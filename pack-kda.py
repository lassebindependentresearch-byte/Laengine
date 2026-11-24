"""Join five identically quantized projections per recurrent BailingMoE3 layer."""
from pathlib import Path
import sys

import gguf
import numpy as np

source, destination = map(Path, sys.argv[1:3])
pack_conv = "--conv" in sys.argv[3:]
pack_experts = "--experts" in sys.argv[3:]
reader = gguf.GGUFReader(source)
arch = reader.get_field('general.architecture').contents()
if arch != 'bailingmoe3':
    raise ValueError(f'Expected bailingmoe3, found {arch}')
writer = gguf.GGUFWriter(destination, arch, endianess=reader.endianess)
for field in reader.fields.values():
    if field.name.startswith('GGUF.') or field.name == 'general.architecture':
        continue
    value = field.contents()
    if value is not None:
        subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, field.types[0], sub_type=subtype)

names = {tensor.name: tensor for tensor in reader.tensors}
groups = {}
omitted = set()
for name in names:
    prefix, separator, _ = name.partition('.ssm_f_a.weight')
    if not separator:
        continue
    projections = tuple(names[f'{prefix}.{key}.weight'] for key in
                        ('attn_q', 'attn_k', 'attn_v', 'ssm_f_a', 'ssm_g_a'))
    first = projections[0]
    if first.data.ndim != 2 or any(t.data.shape != first.data.shape or t.tensor_type != first.tensor_type
                                  for t in projections):
        raise ValueError(f'Projection quantization or dimensions differ at {prefix}')
    groups[name] = (f'{prefix}.ssm_in.weight', projections)
    omitted.update(t.name for t in projections if t.name != name)
if len(groups) != 35:
    raise ValueError(f'Expected 35 recurrent layers, found {len(groups)}')

if pack_conv:
    for prefix in (name.split('.ssm_f_a.weight')[0] for name in list(groups)):
        values = tuple(names[f'{prefix}.ssm_conv1d_{key}.weight'] for key in 'qkv')
        first = values[0]
        if first.data.ndim != 3 or first.data.shape[1] != 1 or any(
                t.data.shape != first.data.shape or t.tensor_type != first.tensor_type for t in values):
            raise ValueError(f'Convolution weights differ at {prefix}')
        groups[first.name] = (f'{prefix}.ssm_conv1d.weight', values)
        omitted.update(t.name for t in values[1:])

if pack_experts:
    for name, gate in names.items():
        if not name.endswith('.ffn_gate_exps.weight'):
            continue
        up = names[name.replace('.ffn_gate_exps.', '.ffn_up_exps.')]
        if gate.data.ndim != 3 or gate.data.shape != up.data.shape or gate.tensor_type != up.tensor_type:
            raise ValueError(f'Expert projections differ at {name}')
        groups[name] = (name.replace('.ffn_gate_exps.', '.ffn_gate_up_exps.'), (gate, up))
        omitted.add(up.name)

layout = []
for name, tensor in names.items():
    if name in omitted:
        continue
    out_name, values = groups.get(name, (name, (tensor,)))
    axis = 1 if out_name.endswith(".ffn_gate_up_exps.weight") else 0
    squeeze = out_name.endswith(".ssm_conv1d.weight")
    shape = list(tensor.data.shape)
    shape[axis] *= len(values)
    if squeeze:
        del shape[1]
    writer.add_tensor_info(out_name, tuple(shape), tensor.data.dtype, tensor.n_bytes * len(values), tensor.tensor_type)
    layout.append((values, axis, squeeze))

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for values, axis, squeeze in layout:
    data = values[0].data if len(values) == 1 else np.concatenate([t.data for t in values], axis=axis)
    if squeeze:
        data = data.squeeze(axis=1)
    writer.write_tensor_data(data, tensor_endianess=reader.endianess)
writer.close()
print(f'Wrote {destination}', flush=True)
