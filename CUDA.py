import torch
print(torch.cuda.is_available())

print(torch.version.cuda)
gpu = torch.device("cuda:0")
print("GPU Device:【{}：{}】".format(gpu.type, gpu.index))
print("Total GPU Count :{}".format(torch.cuda.device_count()))
print("Total CPU Count :{}".format(torch.cuda.os.cpu_count()))
print(torch.version.cuda)
print(torch.__version__)

print(torch.cuda.is_available())