import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

p_160k = '/home/oyasumi/Documents/yy/W3result/potsdam_result/mlphead_lightformer_b2_linear/20231114_203343.log.json'
p_80k = '/home/oyasumi/Documents/yy/W3result/potsdam_result/mlphead_lightformer_b2_linear-80K/20231115_200648.log.json'
v_160k = '/home/oyasumi/Documents/yy/W3result/vaihingen_result/mlphead_lightformer_b2_linear/20231124_121119.log.json'
v_80k = '/home/oyasumi/Documents/yy/W3result/vaihingen_result/mlphead_lightformer_b2_linear-80K/20231116_100832.log.json'
v_240k = '/home/oyasumi/Documents/yy/W3result/vaihingen_result/mlphead_lightformer_b2_linear-240k/20231125_104102.log.json'
p_240k = '/home/oyasumi/Documents/yy/W3result/potsdam_result/mlphead_lightformer_b2_linear-240k/20231126_143124.log.json'

# 读取json文件
with open(p_160k, 'r') as f1:
    pdata_160k = [json.loads(line1) for line1 in f1]
pdata_160k.pop(0)
with open(p_80k, 'r') as f2:
    pdata_80k = [json.loads(line2) for line2 in f2]
pdata_80k.pop(0)
# 读取json文件
with open(v_160k, 'r') as f3:
    vdata_160k = [json.loads(line1) for line1 in f3]
vdata_160k.pop(0)
with open(v_80k, 'r') as f4:
    vdata_80k = [json.loads(line2) for line2 in f4]
vdata_80k.pop(0)
with open(v_240k, 'r') as f5:
    vdata_240k = [json.loads(line3) for line3 in f5]
vdata_240k.pop(0)
with open(p_240k, 'r') as f6:
    pdata_240k = [json.loads(line4) for line4 in f6]
pdata_240k.pop(0)

piters_160k = []
pmious_160k = []
piters_80k = []
pmious_80k = []
viters_160k = []
vmious_160k = []
viters_80k = []
vmious_80k = []
viters_120k = [12000,24000,36000,48000,60000,72000,84000,96000,108000,120000]
vmious_120k = [0.7298,0.7375,0.7573,0.7613,0.7661,0.7789,0.7736,0.7733,0.7759,0.7789]
piters_120k = viters_120k
pmious_120k = [0.7214,0.7404,0.7603,0.7708,0.7944,0.7978,0.8127,0.8086,0.8178,0.8282]
# 提取epoch和miou数据
for i in pdata_160k:
    if 'loss' in i:
        piters_160k.append(i['iter'])
        pmious_160k.append(i['loss'])
for i in pdata_80k:
    if 'loss' in i:
        piters_80k.append(i['iter'])
        pmious_80k.append(i['loss'])

# 提取epoch和miou数据
for i in vdata_160k:
    if 'loss' in i:
        viters_160k.append(i['iter'])
        vmious_160k.append(i['loss'])
for i in vdata_80k:
    if 'loss' in i:
        viters_80k.append(i['iter'])
        vmious_80k.append(i['loss'])

# piters_160k,pmious_160k = smooth_xy(piters_160k, pmious_160k)
# viters_160k,vmious_160k = smooth_xy(viters_160k, vmious_160k)
# piters_80k,pmious_80k = smooth_xy(piters_80k, pmious_80k)
# viters_80k,vmious_80k = smooth_xy(viters_80k, vmious_80k)
# piters_120k,pmious_120k = smooth_xy(piters_120k, pmious_120k)
# viters_120k,vmious_120k = smooth_xy(viters_120k, vmious_120k)
# 绘制曲线图\
plt.plot(piters_160k, pmious_160k, color='red', label='Potsdam-160k')
# plt.plot(piters_120k, pmious_120k, color='pink', label='Potsdam-120k')
plt.plot(piters_80k, pmious_80k,  color='blue', label='Potsdam-80k')
plt.plot(viters_160k, vmious_160k, color='orange', label='Vaihingen-160k')
# plt.plot(viters_120k, vmious_120k, color='purple', label='Vaihingen-120k')
plt.plot(viters_80k, vmious_80k, color='green', label='Vaihingen-80k')


# plt.xticks(x_l, rotation=20)
plt.xlabel('Iterations')
plt.ylabel('loss')
plt.title('loss over iterations on training')
plt.legend()#添加图例
# plt.show()

# 将图像保存到文件中
plt.savefig('/home/oyasumi/Documents/yy/W3result/loss.png')

# iters = [80000, 160000, 240000]
# mious_p = [85.76, 87.81, 86.55]
# mious_v = [78.48, 79.36, 79.04]
# # 绘制曲线图
# # plt.plot(iters, mious_p, color='red', label='Potsdam')
# plt.plot(iters, mious_v, color='blue', label='Vaihingen')
# plt.xticks(iters)
# plt.xlabel('Iterations')
# plt.ylabel('mIoU')
# plt.title('mIoU over iterations')
# # plt.legend()#添加图例
# # plt.show()
#
# # 将图像保存到文件中
# plt.savefig('/home/oyasumi/Documents/yy/W3result/Vaihingen_test.png')