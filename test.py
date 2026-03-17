import torch
print("torch:", torch.__version__)
print("cuda in torch:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("cc:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
x = torch.randn(4, 4, device="cuda")
print(x @ x)
