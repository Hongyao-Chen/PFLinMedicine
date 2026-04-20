import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
# from pyarrow.types import null
from sympy import true
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# 设置全局字体为 Arial
# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

###################################################################################
# loss =  [2.1728620437961066, 2.1010204015858567, 1.965741905747763, 1.8148331863044838, 1.6934061021305489, 1.6188974470052673, 1.4990026015872868, 1.4538254847800012, 1.3855435632903939, 1.316408950127511, 1.268487436925903, 1.2455463343478768, 1.164594104074217, 1.1215187418641877, 1.066354152671174, 1.0243358649959027, 0.96484139709238, 0.9551897589369436, 0.8589921567044901, 0.8418819111176613, 0.7804530552843822, 0.7316865523050943, 0.680479488366752, 0.6458013381711181, 0.5932242812415138, 0.5408797286619929, 0.5107957982582495, 0.478182995790494, 0.45714466694432715, 0.41564897861476635, 0.3963491888557078, 0.3732299419707715, 0.35129036894161497, 0.32889989188409446, 0.3234761888715771, 0.3203639624666567, 0.3047925143324508, 0.3045401537630126, 0.30126916877494275, 0.31804995409824066, 0.34690833219421907, 0.3638643473531183, 0.3605591455601837, 0.3256111219352197, 0.34535089410229236, 0.3653376777535688, 0.36430227511986546, 0.38791711922450894, 0.3643498480307545, 0.3369959379803267, 0.3337484228170817, 0.3223236125280865, 0.34746360233527857, 0.3383093655685469, 0.32685958435660006, 0.2940908733809868, 0.28729998552636976, 0.29132695728070807, 0.270439249640085, 0.2439448502819564, 0.22140871491087735, 0.21169453173979927, 0.19247943171817317, 0.18574726008582687, 0.18021051208021033, 0.1714065363567696, 0.16499650192802057, 0.16269846266740812, 0.15671427009681613, 0.15610799491119673, 0.16193991323165208, 0.16582294034299389, 0.1646850972343976, 0.16601065728582087, 0.18523761138962608, 0.18444077813558896, 0.19998113389568442, 0.20546283229615162, 0.21132593041858652, 0.22136674114822627, 0.20673889303045673, 0.21214040716167848, 0.22057165546382307, 0.22188618683656072, 0.2127792823518537, 0.21399548503606186, 0.2119087578268227, 0.20273866635936738, 0.19640731851141482, 0.19545300448299877, 0.19457020804555578, 0.1861622347903124, 0.18381152043992283, 0.17394498308498788, 0.16999646573196958, 0.1685588710214823, 0.16844508349354406, 0.1647219448540255, 0.15751548964754283, 0.15845837938425325, 0.15722760824042162, 0.15799220147853876, 0.15220811654003302, 0.1494224009295068, 0.1433049396650682, 0.14202160731515315, 0.1378749414490269, 0.13251923010249908, 0.1359660802946866, 0.12841298400063386, 0.1297632231401433, 0.1243243095857613, 0.12574550596826495, 0.12477577659185812, 0.11881041555399835, 0.11859483216286536, 0.11833604896364106, 0.11476440021891084, 0.11535998609492044, 0.11479334843606517, 0.11478838840955415, 0.11491081001385327, 0.1109444402271606, 0.10958455838686071, 0.10534670168985537, 0.10407569695870708, 0.09981826875996756, 0.09775971597427345, 0.09507280146079347, 0.09438422279015951, 0.09459303348145594, 0.09450041412183519, 0.0953977241961319, 0.09403546743767531, 0.0918070026749765, 0.09220976293002434, 0.09077364504423208, 0.08847766988249893, 0.08512990222835372, 0.08573798323699859, 0.0855080600425963, 0.0828549193385848, 0.08139241440341283, 0.08027893356801959, 0.07920623269223104, 0.07774784707302236, 0.07481184809013637, 0.07516602919402139, 0.07217689972054901, 0.07151654430419013, 0.07144946053902933, 0.07195486269599966, 0.07145694014183575, 0.06864670173905853, 0.06675262730170024, 0.06733915041353516, 0.06830641801838047, 0.06671018381241484, 0.06547507700009546, 0.06312561875503607, 0.06377384950578015, 0.06234601962868486, 0.06062227135513329, 0.060093812569567445, 0.058145069561232436, 0.05761223724427859, 0.057955168248685046, 0.055913387910365576, 0.05468080337319916, 0.05425119802654732, 0.0538625668335651, 0.05337648394638058, 0.05396079597134492, 0.05316839544125914, 0.052371029131535254, 0.051574857339372976, 0.05074615674395478, 0.049473178690402776, 0.04864236096838926, 0.04938617536646286, 0.048192406677493214, 0.04665979282014305, 0.047111397039623706, 0.04623529899936278, 0.04534195536346105, 0.0442296109796109, 0.04294142342247833, 0.04263224322230325, 0.041506924035639284, 0.04173533546025856, 0.0413623686225397, 0.0401725903869268, 0.0396251311698607, 0.03918275691892686, 0.03769821848427684, 0.03729135230048506, 0.036912206165160055, 0.03691660672183783, 0.036592455537491214, 0.03582367128885792, 0.035261227316960124, 0.03497650188433514, 0.03436406139062703, 0.03342146963995599, 0.0331869010583325, 0.03334334568835119, 0.03203158733682112, 0.03176382109646115, 0.03294875524120477, 0.031806858487701935, 0.030829616695256225, 0.03039388353704611, 0.029898605426083247, 0.02972691296086757, 0.029906810534623938, 0.03017275337413215, 0.028826912153456067, 0.02888504443522938, 0.027942033262753938, 0.027884394394172674, 0.02761459880928318, 0.026566526518817468, 0.02674695273200809, 0.027119239984717956, 0.02640072767286899, 0.026101162426154796, 0.024902225103367433, 0.02442105313839602, 0.024308144107556447, 0.024360724967114663, 0.02404831760792585, 0.02421020695332262, 0.024456911230478263, 0.02333291876008433, 0.022759071711774364, 0.023133202601780343, 0.022832716378718105, 0.022472446388067807, 0.02165722575572573, 0.02123922812227327, 0.02119001415777368, 0.021072984229907672, 0.020185867703549017, 0.019668204266312647, 0.01917521140837802, 0.019421057498131247, 0.019015816999971936, 0.019944027502892526, 0.019041360267375063, 0.018680798416708926, 0.01868672308310052, 0.019099957721929958, 0.017678345288442147, 0.017395408723460615, 0.016333327959276986, 0.016824050341599794, 0.016519310469224065, 0.01696412227746653, 0.015215205622971439, 0.01567494664313695, 0.016405749209114574, 0.015855912678707455, 0.01684271634165018, 0.015510988613822817, 0.015344251296157204, 0.014318032450941442, 0.016813188712098113, 0.015154467336227622, 0.015038590987699397, 0.014496896686228902, 0.013671620827581598, 0.013354399832199397, 0.014927540862098472, 0.01340478278954047, 0.013182319280170149, 0.011314978406335281, 0.011580754569917107, 0.011505470661092431, 0.012858494091814193, 0.0110727972081631, 0.009914274614494241, 0.009641475574560071, 0.01012857922840341, 0.013940124224588967, 0.017285065946680344, 0.01258791902925064, 0.014535167762872765, 0.010200216995369743, 0.009569671458365353, 0.014018342454057139, 0.009819168597997906, 0.009728281989251776, 0.014831561792387177, 0.00893194161304378, 0.008127776298852418, 0.01144657585581411, 0.008279559133170735, 0.00902573889740801, 0.008294303254921455, 0.0096701680457963, 0.009058708242675165]
# acc =[0.16718788577071847, 0.22747099004197185, 0.2749156448028969, 0.291251748827257, 0.3582009711134886, 0.3719858447864373, 0.39033824376594517, 0.407209283186569, 0.4699613200559625, 0.4851041066578882, 0.5140729158093984, 0.5268290675664554, 0.558596000329191, 0.6061229528433874, 0.6602337256192906, 0.6745946835651386, 0.7120813101802321, 0.7040161303596412, 0.756069459303761, 0.7557814171673113, 0.789523496008559, 0.8052012180067484, 0.8242119990124269, 0.8336762406386306, 0.8436342687844622, 0.8579540778536746, 0.8699695498312896, 0.8763476257098182, 0.8743313307546704, 0.8860587605958358, 0.8913669656818369, 0.895193811208954, 0.8995555921323348, 0.9022302691136532, 0.9033001399061805, 0.9044111595753436, 0.9088552382519958, 0.908443749485639, 0.9092255781417168, 0.9095136202781664, 0.904863797218336, 0.9055633281211423, 0.9063451567772199, 0.9071681343099334, 0.909184429265081, 0.9096370669080734, 0.9094724714015308, 0.9083203028557321, 0.9092667270183524, 0.9071681343099334, 0.906962389926755, 0.9129701259155625, 0.9098839601678874, 0.9077030697061971, 0.9136696568183689, 0.9149864208707102, 0.9152333141305242, 0.9132170191753766, 0.9152744630071599, 0.9208707102296108, 0.9272076372315036, 0.9271253394782323, 0.93161056703152, 0.9314048226483417, 0.930746440622171, 0.9337914574932105, 0.9339149041231174, 0.9351082215455518, 0.938317833923134, 0.9363015389679862, 0.9335034153567607, 0.930746440622171, 0.9294708254464653, 0.928647847913752, 0.9178256933585713, 0.9181548843716566, 0.908443749485639, 0.906962389926755, 0.9042054151921652, 0.9021891202370176, 0.9076207719529257, 0.9053987326145996, 0.9017776314706608, 0.9000905275285984, 0.9036704797959015, 0.9025183112501028, 0.9042054151921652, 0.9082791539790964, 0.9107480865772365, 0.9111184264669574, 0.9111184264669574, 0.9142457410912682, 0.9163443337996873, 0.9201300304501687, 0.9223932186651305, 0.9227635585548515, 0.9213644967492387, 0.9237099827174718, 0.9275779771212246, 0.9282363591473953, 0.9298411653361863, 0.9296765698296436, 0.931980906921241, 0.9325569911941404, 0.9341206485062957, 0.9338737552464818, 0.9361780923380791, 0.9380297917866842, 0.9367130277343428, 0.9397168957287466, 0.9403341288782816, 0.9428030614764217, 0.940663319891367, 0.9411571064109949, 0.943749485639042, 0.9446959098016624, 0.9430911036128714, 0.9445313142951197, 0.9449839519381121, 0.9445724631717554, 0.9433379968726854, 0.9427207637231504, 0.9444901654184841, 0.9458480783474611, 0.9468356513867171, 0.9479055221792445, 0.9495103283680355, 0.9505801991605629, 0.9502098592708419, 0.95070364579047, 0.9509916879269196, 0.9508270924203769, 0.9502921570241133, 0.9503744547773846, 0.9511974323100979, 0.9512385811867337, 0.9516089210764546, 0.9521850053493539, 0.9529668340054317, 0.9522673031026253, 0.9528845362521603, 0.9532548761418813, 0.9538721092914163, 0.953666364908238, 0.9535840671549667, 0.954160151427866, 0.955147724467122, 0.9545716401942227, 0.955764957616657, 0.9561764463830137, 0.9561764463830137, 0.9548596823306724, 0.955147724467122, 0.9567936795325488, 0.9564233396428278, 0.9567936795325488, 0.9560941486297424, 0.9558884042465641, 0.9564644885194634, 0.9575755081886265, 0.9568759772858201, 0.9579458480783475, 0.9585219323512468, 0.9588511233643322, 0.9589745699942391, 0.9586453789811538, 0.9581515924615258, 0.9594683565138672, 0.9593449098839602, 0.9592214632540532, 0.9587276767344252, 0.9596741008970455, 0.9587688256110608, 0.9588099744876964, 0.9581104435848902, 0.9588922722409678, 0.9587276767344252, 0.9593860587605958, 0.9586453789811538, 0.9576578059418978, 0.9586453789811538, 0.9595095053905028, 0.9592626121306889, 0.9594272076372315, 0.9588922722409678, 0.9584807834746112, 0.9591391655007818, 0.9588099744876964, 0.9598386964035882, 0.9593449098839602, 0.9593860587605958, 0.9599209941568595, 0.9603324829232162, 0.9597563986503168, 0.9601678874166735, 0.9609085671961155, 0.9600855896634022, 0.9602501851699449, 0.9600444407867665, 0.9599209941568595, 0.9599621430334951, 0.9603736317998519, 0.9607439716895729, 0.9604147806764876, 0.9606205250596659, 0.9604559295531232, 0.9604147806764876, 0.9607851205662086, 0.9594683565138672, 0.9601267385400378, 0.9595506542671385, 0.9595506542671385, 0.9603736317998519, 0.9604147806764876, 0.9599209941568595, 0.9596741008970455, 0.9602501851699449, 0.9595095053905028, 0.9602501851699449, 0.9600444407867665, 0.9599209941568595, 0.9597152497736812, 0.9606205250596659, 0.9584396345979754, 0.9584807834746112, 0.9580692947082544, 0.9594683565138672, 0.9588511233643322, 0.9593037610073245, 0.9581927413381615, 0.9590157188708748, 0.9582338902147971, 0.9585219323512468, 0.9591391655007818, 0.9592214632540532, 0.9591391655007818, 0.9590568677475105, 0.9586865278577895, 0.9590980166241462, 0.9593449098839602, 0.9590980166241462, 0.9588099744876964, 0.9599621430334951, 0.9597975475269526, 0.9596741008970455, 0.9585219323512468, 0.9598386964035882, 0.9589334211176035, 0.9599209941568595, 0.9597152497736812, 0.9600032919101309, 0.9593037610073245, 0.9602501851699449, 0.9587688256110608, 0.9602501851699449, 0.9593860587605958, 0.9596329520204099, 0.9586453789811538, 0.9602501851699449, 0.9586453789811538, 0.9590568677475105, 0.9586453789811538, 0.9593037610073245, 0.9581515924615258, 0.9592626121306889, 0.9585630812278825, 0.9585219323512468, 0.9588099744876964, 0.9590568677475105, 0.9576166570652621, 0.9586453789811538, 0.9580281458316188, 0.9576166570652621, 0.9580692947082544, 0.9580692947082544, 0.9586042301045181, 0.9583161879680685, 0.9585219323512468, 0.9585219323512468, 0.9583573368447041, 0.9588511233643322, 0.9583161879680685, 0.9590157188708748, 0.9575755081886265, 0.9570817216689984, 0.9584807834746112, 0.9565879351493705, 0.9581104435848902, 0.9587276767344252, 0.9572051682989055, 0.9580692947082544, 0.9584807834746112, 0.9563821907661921, 0.9588099744876964, 0.9579869969549831, 0.9574520615587194, 0.9574520615587194, 0.9567936795325488, 0.9586042301045181, 0.9563410418895564, 0.9576989548185334]
#
# # # 创建图形和第一个Y轴
# fig, ax1 = plt.subplots(figsize=(25, 4))
#
# # 绘制第一条折线（左侧Y轴）
# color1 = 'tab:blue'
#
# line1 = ax1.plot(loss, color=color1, linewidth=2, label='Loss')
# ax1.tick_params(axis='y', labelcolor=color1,labelsize=15)
# ax1.tick_params(axis='x',labelsize=24)
# # 创建第二个Y轴（右侧）
# ax2 = ax1.twinx()
# color2 = 'tab:red'
# line2 = ax2.plot(acc, color=color2, linestyle='--', linewidth=2, label='Accuracy')
# ax2.tick_params(axis='y', labelcolor=color2,labelsize=15)
#
# # # 添加标题和图例
#
# # # 合并图例
# # lines = line1 + line2
# # labels = [l.get_label() for l in lines]
# # ax1.legend(lines, labels, loc='upper left', fontsize=24)
# # ax1.tick_params(axis='both', width=4)  # 同时加粗x和y轴
# ax1.grid(True, axis='x', alpha=0.7,linewidth=4)  # 主坐标轴
# ax2.grid(True, axis='y', alpha=0.7,linewidth=2)  # 次坐标轴
# # 调整布局
# plt.tight_layout()
# plt.grid(True, which='both', axis='both', alpha=0.7)
# plt.show()
# import matplotlib.pyplot as plt
#
# ###################################################################################
import matplotlib.pyplot as plt
import numpy as np

# 两组数据
data0 = [0.7770562770562771, 0.7101769911504425, 0.7497565725413826,
         0.7510917030567685, 0.7303877366997295, 0.7302504816955684, 0.77]
data2 = [0.7987722943722944, 0.7831858407079646, 0.7667964946445959,
         0.7991266375545851, 0.7475100090171325, 0.7461040462427746, 0.7803333333333333]
data1 = [.7303, .7303, .7303,
         .7303, .7303, .7303, .7303]

# 两组数据对应的标准差（这里使用模拟值，你需要替换为实际计算的标准差）
std0 = [0.010, 0.002, 0.004, 0.007, 0.009, 0.007, 0.005]  # 第一组数据的标准差
std2 = [0.006, 0.007, 0.012, 0.006, 0.003, 0.004, 0.002]  # 第二组数据的标准差
std1 = [0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]  # 第二组数据的标准差

# X轴标签
x_labels = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']

# 设置柱状图位置
x = np.arange(len(x_labels))  # 标签位置
width = 0.6  # 柱状图宽度

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制柱状图
# 绘制柱状图
rects0 = ax.bar(x - width / 3, data0, width / 3, label='Local',
                color='#3498DB', edgecolor='black', linewidth=1.2)
rects1 = ax.bar(x, data1, width / 3, label='FedAvg',
                color='#f58231', edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x + width / 3, data2, width / 3, label='PFL-AC',
                color='#E399A9', edgecolor='black', linewidth=1.2)

# 添加误差线 - 修正了数据匹配问题
ax.errorbar(x - width / 3, data0, yerr=std0, fmt='none',
            ecolor='#333333', capsize=4, capthick=1.5, elinewidth=1.5)
ax.errorbar(x, data1, yerr=std1, fmt='none',
            ecolor='#333333', capsize=4, capthick=1.5, elinewidth=1.5)
ax.errorbar(x + width / 3, data2, yerr=std2, fmt='none',
            ecolor='#333333', capsize=4, capthick=1.5, elinewidth=1.5)

# 添加标题和标签
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

# 设置Y轴范围，使图形更美观
ax.set_ylim(0.65, 0.85)

# 添加数值标签
# 调整布局
fig.tight_layout()

# 显示图形
plt.show()
# ###############################################################

# 数据
# count = {
#     'H1 \n (human breast cancer) ': {'1': 2327, '2': 1366},
#     'H2 \n (canine lung cancer)': {'1': 855, '2': 951},
#     'H3 \n (canine lymphosarcoma)': {'1': 3959, '2': 4257},
#     'H4 \n (canine cutaneous mast cell tumor)': {'1': 1283, '2': 2375},
#     'H5 \n (human neuroendocrine tumor)': {'1': 1721, '2': 2714},
#     'H6 \n (canine soft tissue sarcoma)': {'1': 1150, '2': 925},
#     'H7 \n (human melanoma)': {'1': 639, '2': 1761},
# # }
# count = {
#     'H1': {'1': 2327, '2': 1366},
#     'H2': {'1': 855, '2': 951},
#     'H3': {'1': 3959, '2': 4257},
#     'H4': {'1': 1283, '2': 2375},
#     'H5': {'1': 1721, '2': 2714},
#     'H6': {'1': 1150, '2': 925},
#     'H7': {'1': 639, '2': 1761},
# }
#
#
# labels = list(count.keys())
# mitotic = [count[tt]['1'] for tt in labels]
# non_mitotic = [count[tt]['2'] for tt in labels]
#
#
# plt.figure(figsize=(8, 6))
#
#
# plt.barh(
#     labels,
#     mitotic,
#     label='Mitotic',
#     color='#3498DB',  # 更换为较浅的蓝色
#     edgecolor='black',  # 黑色边框
#     linewidth=1.5  # 边框线宽
# )
#
# # 第二组：Non-mitotic（深橙色带黑边，堆叠在第一组右侧）
# plt.barh(
#     labels,
#     non_mitotic,
#     left=mitotic,
#     label='Non-mitotic',
#     color='#E67E22',  # 更换为橙色
#     edgecolor='black',  # 黑色边框
#     linewidth=1.5  # 边框线宽
# )
#
# # 添加图例、标签和标题（可选）
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.title('Image Count for MIDOG++', fontsize=18)
# plt.xlabel('Image Count', fontsize=18)
# plt.ylabel('Tumor Type', fontsize=18)

#
# plt.yticks(fontsize=18)
#
#
# plt.xticks(fontsize=18)
#
#
# plt.legend(
#     title='Cell Division Type',
#     loc='upper right'
# )
#
# plt.tight_layout()
# plt.savefig('2.png', dpi=300)
# plt.show()
##################################################################################
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 1. 把你的 statistic 粘过来
# statistic =[[(0, 2327), (1, 1366)], [(0, 855), (1, 951)], [(0, 3959), (1, 4257)], [(0, 1286), (1, 2375)], [(0, 1721), (1, 2714)], [(0, 1150), (1, 925)], [(0, 639), (1, 1761)]]
#
# # 2. 构造“行=client，列=类别”的矩阵
# # 2. 构造矩阵（行=client，列=类别）
# num_clients = len(statistic)
# max_class = max(cls for stat in statistic for cls, _ in stat)
# num_classes = max_class + 1
#
# mat = np.zeros((num_clients, num_classes), dtype=int)
# for c, stat in enumerate(statistic):
#     for cls, cnt in stat:
#         mat[c, cls] = cnt
#
# # 3. 画热力图
# plt.figure(figsize=(12, 10))
# ax = sns.heatmap(mat.T,
#                  annot=False,
#                  cmap='YlGnBu',
#                  linewidths=.0,
#                  cbar_kws={'label': 'Data Size'})
# ax.figure.axes[-1].yaxis.label.set_size(18)   # color-bar 字体
# cbar = ax.collections[0].colorbar          # 拿到 colorbar 对象
# cbar.ax.tick_params(labelsize=18)          # 放大刻度数字
# ax.set_xlabel('Clients', fontsize=18)
# ax.set_ylabel('Labels', fontsize=18)
# ax.set_title('Label Distribution Across Clients', fontsize=18)
#
# # 关键：放大刻度字体
# ax.tick_params(axis='both', labelsize=18)
#
# plt.tight_layout()
# # plt.savefig('heatmap_client_class.png', dpi=300)
# plt.show()
# ######################################################
# import matplotlib.pyplot as plt
#
# # 数据
# data = {
#     'Hospital 1 \n (human breast cancer)': {
#         '1': 451 + 582 + 688,  # Mitotic
#         '2': 724 + 1066 + 924   # Non-mitotic
#     }
# }
#
# categories = ['Mitotic', 'Non-mitotic']
# values = [data['Hospital 1 \n (human breast cancer)']['1'],
#           data['Hospital 1 \n (human breast cancer)']['2']]
#
# # 绘制柱状图
# plt.figure(figsize=(8, 6))
# bars = plt.bar(categories, values, color=['#FF9999', '#99FF99'])
#
# # 添加数据标签
# for bar in bars:
#     height = bar.get_height()
#
# plt.xticks(fontsize=18)
# # plt.title('Image Count for Human Breast Cancer', fontsize=16)
# # plt.xlabel('Cell Division Type', fontsize=14)
# # plt.ylabel('Image Count', fontsize=14)
# plt.grid(True, linestyle='--', alpha=1)
#
# plt.show()
# #####################################################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
#
# # 设置参数和数据
# x_cnt = 7
# y_cnt = 7
#
# raw_data = [
#     # [0.7770562770562771, 0.7101769911504425, 0.7497565725413826, 0.7510917030567685, 0.7303877366997295,
#     #  0.7302504816955684, 0.77],
#     [0.7770562770562771],
#     [0.7835497835497836, 0.7389380530973452],
#     [0.788961038961039, 0.7433628318584071, 0.7653359298928919],
#     [0.7904545454545454, 0.7411504424778761, 0.7626016553067186, 0.7838427947598253],
#     [0.7932900432900433, 0.7654867256637168, 0.7629913339824732, 0.7903930131004366, 0.7348963029756538],
#     [0.7922077922077922, 0.7699115044247787, 0.7667964946445959, 0.8034934497816594, 0.7457168620378719,
#      0.7418111753371869],
#     [0.7987722943722944, 0.7831858407079646, 0.7667964946445959, 0.7991266375545851, 0.7475100090171325,
#      0.7461040462427746, 0.7803333333333333]
# ]
#
# # 创建数据矩阵
# data_matrix = np.full((x_cnt, y_cnt), np.nan)
# for i in range(len(raw_data)):
#     for j in range(len(raw_data[i])):
#         if j < y_cnt:
#             data_matrix[i, j] = raw_data[i][j]
#
# # 创建标签
# x_labels = [ 'H1', 'H1~H2', 'H1~H3', 'H1~H4', 'H1~H5', 'H1~H6', 'H1~H7']
# y_labels = [f'H{i + 1}' for i in range(y_cnt)]
#
# # 创建图形和坐标轴
# fig, ax = plt.subplots(figsize=(10, 8))
#
# # 创建自定义颜色映射
# colors = ['#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
# cmap = LinearSegmentedColormap.from_list("custom_reds", colors, N=100)
#
# # 绘制热力图
# im = ax.imshow(data_matrix, cmap=cmap, vmin=0.73, vmax=0.81, aspect='auto')
#
# # 设置坐标轴
# ax.set_xticks(np.arange(len(y_labels)))
# ax.set_yticks(np.arange(len(x_labels)))
# ax.set_xticklabels(y_labels, fontfamily='Arial')
# ax.set_yticklabels(x_labels, fontfamily='Arial')
#
# # 添加数值标签（只在下三角区域显示）
# for i in range(x_cnt):
#     for j in range(y_cnt):
#         if not np.isnan(data_matrix[i, j]):
#             text = ax.text(j, i, f'{data_matrix[i, j]:.4f}',
#                            ha="center", va="center",
#                            color="black",
#                            fontfamily="Arial",
#                            fontsize=18)
#
# # 添加颜色条
# cbar = fig.colorbar(im, ax=ax, shrink=0.8)
#
# cbar.ax.tick_params(labelsize=18)
# for label in cbar.ax.get_xticklabels():
#     label.set_fontfamily('Arial')
#
# # 设置标题和标签
# #ax.set_xlabel('Personalised Performance', fontfamily='Arial', fontsize=18)
# #ax.set_ylabel('Participating Hospitals', fontfamily='Arial', fontsize=18)
#
# plt.tight_layout()
# # 保存为PDF格式
# plt.savefig('1.png',
#            format='png',
#            dpi=300,
#            bbox_inches='tight',
#            facecolor='white',
#            edgecolor='none',
#            transparent=False)
#
# plt.show()
#######################################################################
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import numpy as np
#
# # 您的数据
# bbox_data = [
#     # {"bbox": [3602, 636, 3652, 686], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1663},
#     # {"bbox": [4446, 409, 4496, 459], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1664},
#     # {"bbox": [4343, 1053, 4393, 1103], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1665},
#     # {"bbox": [1203, 2528, 1253, 2578], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1666},
#     # {"bbox": [5293, 2566, 5343, 2616], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1667},
#     # {"bbox": [4583, 3631, 4633, 3681], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1668},
#     # {"bbox": [4306, 3927, 4356, 3977], "category_id": 1, "image_id": 69, "labels": [2, 1, 1], "id": 1669},
#     {"bbox": [4307, 4534, 4357, 4584], "category_id": 1, "image_id": 69, "labels": [2, 1, 1], "id": 1670},
#     {"bbox": [4814, 4348, 4864, 4398], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1671},
#     # {"bbox": [1491, 5253, 1541, 5303], "category_id": 2, "image_id": 69, "labels": [2, 2], "id": 1672}
# ]
#
# # 加载TIFF图像
# try:
#     image_path = r"D:\FLProjects\PFLlib-master\dataset\MIDOGpp\images\069.tiff"  # 请确保图像路径正确
#     image = Image.open(image_path)
#     image = np.array(image)
#
#     # 创建图形
#     fig, ax = plt.subplots(1, 1, figsize=(15, 10))
#
#     # 显示图像
#     ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
#
#     # 定义颜色映射（根据category_id）
#     color_map = {1: 'red', 2: 'blue'}
#
#     # 绘制每个边界框
#     for item in bbox_data:
#         bbox = item['bbox']  # [x_min, y_min, x_max, y_max]
#         category_id = item['category_id']
#         box_id = item['id']
#
#         # 计算矩形的宽度和高度
#         width = bbox[2] - bbox[0]
#         height = bbox[3] - bbox[1]
#
#         # 创建矩形补丁
#         rect = patches.Rectangle(
#             (bbox[0], bbox[1]), width, height,
#             linewidth=2, edgecolor=color_map[category_id],
#             facecolor='none',  # 不填充
#             label=f'Category {category_id}'
#         )
#
#         # 添加矩形到坐标轴
#         ax.add_patch(rect)
#
#         # # 添加框的ID标签
#         # ax.text(bbox[0], bbox[1] - 10, f'ID:{box_id}',
#         #         fontsize=8, color=color_map[category_id])
#
#     # 设置图形属性
#     # ax.set_title('Bounding Boxes on 069.tiff\n(Red: Category 1, Blue: Category 2)')
#     ax.grid(False)
#
#     # 添加图例（去重）
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     # ax.legend(by_label.values(), by_label.keys(), loc='upper right')
#
#     plt.tight_layout()
#     plt.show()
#
# except FileNotFoundError:
#     print(f"错误：找不到图像文件 {image_path}")
#     print("请确保：")
#     print("1. 图像文件 '069.tiff' 存在于当前工作目录")
#     print("2. 或者提供完整的图像文件路径")
# except Exception as e:
#     print(f"加载图像时发生错误: {str(e)}")
###############################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 16组数据准备（示例，需替换为你的实际数据）
# 每组数据格式：{'Algorithm': [...], 'Performance': [...], 'Std': [...]}
# --------------------------
all_data = [
    # 第1组数据（示例）
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL-AC', 'Lay-PFL-AC'],
        'Performance': np.array([93.24, 96.11, 96.32, 96.18, 94.76, 95.25, 95.89, 96.40, 96.09, 96.36]),
        'Std': np.array([0.35, 0.03, 0.02, 0.16, 0.07, 0.07, 0.31, 0.05, 0.13, 0.09])
    },
    # 第2组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([92.87, 95.13, 95.63, 95.68, 92.82, 93.54, 95.12, 95.66, 96.15, 96.64]),
        'Std': np.array([0.10, 0.05, 0.14, 0.05, 0.07, 0.06, 0.05, 0.04, 0.11, 0.04])
    },
    # 第3组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([96.01, 97.87, 97.76, 97.90, 97.36, 97.56, 97.60, 97.70, 97.86, 98.05]),
        'Std': np.array([0.27, 0.07, 0.01, 0.04, 0.13, 0.11, 0.08, 0.08, 0.02, 0.07])
    },
    # 第4组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([95.44, 96.19, 96.64, 96.56, 96.04, 95.59, 96.79, 97.01, 97.53, 97.38]),
        'Std': np.array([0.19, 0.05, 0.21, 0.04, 0.07, 0.30, 0.09, 0.04, 0.04, 0.06])
    },
    # 第1组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([95.05, 97.41, 97.68, 97.78, 97.11, 97.34, 97.76, 97.63, 97.92, 97.99]),
        'Std': np.array([0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.01, 0.09, 0.01, 0.02])
    },
    # 第2组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([88.29, 90.52, 90.63, 92.00, 88.30, 89.31, 90.82, 91.18, 91.67, 92.21]),
        'Std': np.array([0.13, 0.00, 0.07, 0.11, 0.13, 0.17, 0.20, 0.02, 0.13, 0.05])
    },
    # 第3组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([41.75, 51.43, 59.62, 54.65, 45.19, 47.58, 55.79, 56.71, 60.31, 61.17]),
        'Std': np.array([0.12, 0.03, 0.21, 0.16, 0.13, 0.42, 0.08, 0.08, 0.04, 0.02])
    },
    # 第4组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([27.37, 42.36, 43.51, 44.35, 31.64, 33.89, 42.09, 44.83, 45.68, 48.06]),
        'Std': np.array([0.16, 0.01, 0.11, 0.07, 0.29, 0.14, 0.09, 0.09, 0.11, 0.13])
    },
    # 第5组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([93.94, 95.35, 93.07, 95.57, 94.47, 94.70, 95.44, 95.40, 95.78, 95.86]),
        'Std': np.array([0.08, 0.03, 0.16, 0.19, 0.10, 0.13, 0.07, 0.12, 0.05, 0.03])
    },
    # 第6组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([97.68, 93.92, 95.20, 97.80, 95.62, 90.24, 97.38, 97.64, 98.52, 98.66]),
        'Std': np.array([0.12, 0.09, 0.01, 0.12, 0.03, 0.04, 0.13, 0.21, 0.04, 0.02])
    },
    # 第7组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([69.99, 61.33, 62.77, 61.17, 63.37, 55.95, 67.19, 67.95, 71.86, 72.89]),
        'Std': np.array([0.16, 0.15, 0.22, 0.18, 0.47, 0.19, 0.31, 0.06, 0.08, 0.12])
    },
    # 第8组数据
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo',
                      'FedAMP', 'FedPHP', 'FedALA', 'PFL_AC_Server', 'Lay-PFL_AC_Server'],
        'Performance': np.array([91.88, 95.79, 94.02, 93.53, 92.55, 95.42, 94.50, 91.09, 97.25, 98.07]),
        'Std': np.array([0.51, 0.39, 0.41, 0.42, 0.33, 0.17, 0.64, 0.20, 0.43, 0.09])
    },
    # 第2组数据 (K=50)
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo', 'FedAMP', 'FedPHP', 'FedALA', 'Ours',
                      'Ours+'],
        'Performance': np.array([43.55, 48.07, 54.99, 40.06, 42.45, 44.71, 57.27, 59.31, 61.66, 62.72]),
        'Std': np.array([0.61, 0.32, 0.12, 0.21, 0.28, 0.08, 0.43, 0.11, 0.26, 0.07])
    },
    # 第3组数据 (K=100)
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo', 'FedAMP', 'FedPHP', 'FedALA', 'Ours',
                      'Ours+'],
        'Performance': np.array([39.86, 43.83, 49.49, 35.76, 38.69, 41.51, 51.39, 54.43, 60.34, 61.65]),
        'Std': np.array([0.43, 0.09, 0.17, 0.21, 0.29, 0.10, 0.65, 0.06, 0.19, 0.13])
    },
    # 第5组数据 (K=50|100)
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo', 'FedAMP', 'FedPHP', 'FedALA', 'Ours',
                      'Ours+'],
        'Performance': np.array([39.72, 43.09, 48.57, 35.13, 38.38, 41.43, 38.34, 54.42, 59.76, 60.08]),
        'Std': np.array([0.31, 0.17, 0.05, 0.35, 0.42, 0.09, 0.32, 0.11, 0.26, 0.08])
    },
    # 第6组数据 (K=25|100)
    {
        'Algorithm': ['Per-FedAvg', 'FedRep', 'FedCP', 'FedAS', 'FedFomo', 'FedAMP', 'FedPHP', 'FedALA', 'Ours',
                      'Ours+'],
        'Performance': np.array([39.41, 42.23, 46.38, 33.15, 38.42, 40.88, 31.49, 53.18, 53.54, 54.80]),
        'Std': np.array([0.54, 0.33, 0.23, 0.61, 0.09, 0.02, 0.23, 0.12, 0.09, 0.21])
    }
]

# --------------------------
# 创建4×4画布
# --------------------------
fig, axes = plt.subplots(4, 4, figsize=(40, 32))  # 4×4子图，总尺寸放大以保证清晰度
axes = axes.flatten()  # 将2D数组转为1D，方便循环

# --------------------------
# 为每个子图绘图
# --------------------------
for idx, data in enumerate(all_data[:16]):  # 确保只取前16组数据
    ax = axes[idx]
    df = pd.DataFrame(data)

    # 生成算法专属颜色
    num_algorithms = len(df['Algorithm'])
    colors = plt.cm.tab10(np.linspace(0, 1, num_algorithms))

    # 绘制散点图和误差线（Performance和Std已转为数组，直接使用）
    for i, (algo, perf, std, color) in enumerate(zip(
            df['Algorithm'],
            0.01 * df['Performance'],  # 归一化到0-1范围
            0.01 * df['Std'],  # 误差线同步归一化
            colors
    )):
        # 散点图
        ax.scatter(
            i, perf,
            s=300,  # 子图中适当减小点大小
            color=color,
            edgecolors='black',
            linewidth=3,
            zorder=3,
        )
        # 误差线（与散点同色）
        ax.errorbar(
            i, perf,
            yerr=std,
            fmt='',
            color=color,
            capsize=3,
            capthick=1.5,
            elinewidth=3,
            alpha=0.8
        )

    # 子图设置
    ax.tick_params(axis='x', which='both', length=0)  # 去掉x轴刻度线
    ax.set_xticks([])  # 去掉x轴标签
    ax.grid(True, alpha=0.6, axis='y')

    # # 可选：为子图添加标题（如数据组别）
    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=8,  # 严格显示5个刻度
            integer=False,  # 允许非整数
            symmetric=False,  # 不强制对称分布
        )
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  # 保留2位小数
    ax.tick_params(
        axis='y',  # 只针对y轴
        labelsize=24,  # 字体大小（可调整）
    )
    for spine in ax.spines.values():  # 获取子图的四个边框（上、下、左、右）
        spine.set_linewidth(2.5)  # 边框粗细（默认约1，2.5较明显）
# --------------------------
# 调整布局：预留顶部图例空间+上下子图间距
# --------------------------
# 生成图例元素（每个算法对应一个散点标记）
algorithms = all_data[0]['Algorithm']
num_algorithms = len(algorithms)
colors = plt.cm.tab10(np.linspace(0, 1, num_algorithms))
legend_elements = [
    Line2D(
        [0], [0], marker='o', color='w',  # 线隐藏，仅显示标记
        markerfacecolor=colors[i], markeredgecolor='black',
        markersize=24, markeredgewidth=3,  # 匹配散点大小
        label=algorithms[i]
    ) for i in range(num_algorithms)
]

# 添加全局图例
fig.legend(
    handles=legend_elements,  # 使用手动创建的元素
    fontsize=32,
    ncol=10,
    bbox_to_anchor=(0.5, 0.99),
    loc='center',
    frameon=False,
)

# --------------------------
# 核心：强制设置上下子图间距
# --------------------------
plt.subplots_adjust(
    top=0.95,  # 顶部留白（为图例）
    bottom=0.02,  # 底部留白
    left=0.05,
    right=0.95,
    hspace=0.15,  # 上下子图间距（关键参数，0.4足够明显，可增大至0.5）
    wspace=0.2  # 左右子图间距（辅助调整）
)
plt.savefig('Result2.png', dpi=300)
plt.show()

#######################################################################
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from PIL import Image
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from tqdm import tqdm
# import seaborn as sns
#
#
# def lda_14_categories_visualization():
#     # 1. 数据路径设置
#     data_path = r"D:\FLProjects\PFLlib-master\dataset\MIDOGpp"  # 添加r前缀避免转义问题
#     tissue_types = [
#         'canine_cutaneous_mast_cell_tumor',
#         'canine_lung_cancer',
#         'canine_lymphosarcoma',
#         'canine_soft_tissue_sarcoma',
#         'human_breast_cancer',
#         'human_melanoma',
#         'human_neuroendocrine_tumor'
#     ]
#     class_names = ['mitotic_figure', 'not_mitotic_figure']
#
#     # 2. 特征提取函数
#     def extract_features(image_path, model):
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image).unsqueeze(0)
#
#         with torch.no_grad():
#             features = model(image)
#
#         return features.squeeze().numpy()
#
#     # 3. 加载模型
#     model = models.resnet50(pretrained=True)
#     model = torch.nn.Sequential(*(list(model.children())[:-1]))
#     model.eval()
#
#     # 4. 收集数据
#     print("📁 收集图片数据...")
#     image_paths = []
#     labels = []  # 细胞类别
#     tissue_labels = []  # 组织类型
#     combined_labels = []  # 组合标签：组织_细胞
#
#     for tissue in tissue_types:
#         tissue_path = os.path.join(data_path, tissue)
#         if not os.path.exists(tissue_path):
#             print(f"⚠️ 路径不存在: {tissue_path}")
#             continue
#
#         for class_name in class_names:
#             c = 0
#             class_path = os.path.join(tissue_path, class_name)
#             if not os.path.exists(class_path):
#                 print(f"⚠️ 路径不存在: {class_path}")
#                 continue
#
#             png_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
#             l = len(png_files)
#             for img_file in png_files:
#                 c += 1
#                 image_paths.append(os.path.join(class_path, img_file))
#                 labels.append(class_name)
#                 tissue_labels.append(tissue)
#                 combined_labels.append(f"{tissue}_{class_name}")  # 组合标签
#                 if c > l * 0.1:
#                     break
#
#     print(f"✅ 找到 {len(image_paths)} 张图片")
#
#     if len(image_paths) == 0:
#         print("❌ 未找到图片")
#         return
#
#     # 5. 提取特征
#     print("🔍 提取特征...")
#     features_list = []
#     valid_combined_labels = []
#     valid_labels = []
#     valid_tissues = []
#
#     for i, img_path in enumerate(tqdm(image_paths)):
#         try:
#             features = extract_features(img_path, model)
#             features_list.append(features.flatten())
#             valid_combined_labels.append(combined_labels[i])
#             valid_labels.append(labels[i])
#             valid_tissues.append(tissue_labels[i])
#         except Exception as e:
#             print(f"⚠️ 处理图片失败: {img_path}, 错误: {e}")
#             continue
#
#     features_array = np.array(features_list)
#     print(f"📊 特征维度: {features_array.shape}")
#
#     # 6. 数据预处理
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features_array)
#
#     # 编码标签 - 使用组合标签进行LDA
#     le_combined = LabelEncoder()
#     y_combined = le_combined.fit_transform(valid_combined_labels)  # 14个组合类别编码
#
#     # 7. LDA分析 - 按14个组合类别进行
#     print("\n🎯 LDA分析: 14个组合类别 (7组织类型 × 2细胞类别)")
#     lda_combined = LinearDiscriminantAnalysis(n_components=2)
#     features_lda_combined = lda_combined.fit_transform(features_scaled, y_combined)
#
#     print(f"📐 LDA解释方差比例: {lda_combined.explained_variance_ratio_.sum():.2%}")
#
#     # 获取所有组合类别名称
#     combined_class_names = le_combined.inverse_transform(range(len(le_combined.classes_)))
#
#     print(f"🔍 实际找到的组合类别: {combined_class_names}")
#     for combined_name in combined_class_names:
#         count = valid_combined_labels.count(combined_name)
#         print(f"  {combined_name}: {count} 个样本")
#
#     print("\n🎯 3D LDA分析: 14个组合类别")
#     lda_3d = LinearDiscriminantAnalysis(n_components=3)  # 使用3个组件
#     features_lda_3d = lda_3d.fit_transform(features_scaled, y_combined)
#
#     print(f"📐 LDA解释方差比例: {lda_3d.explained_variance_ratio_.sum():.2%}")
#     print(f"📊 各组件解释方差: {lda_3d.explained_variance_ratio_}")
#
#     # 8. 3D可视化
#     fig = plt.figure(figsize=(16, 12))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 颜色定义
#     tissue_colors = {
#         'canine_cutaneous_mast_cell_tumor': '#1f77b4',
#         'canine_lung_cancer': '#ff7f0e',
#         'canine_lymphosarcoma': '#2ca02c',
#         'canine_soft_tissue_sarcoma': '#d62728',
#         'human_breast_cancer': '#9467bd',
#         'human_melanoma': '#8c564b',
#         'human_neuroendocrine_tumor': '#e377c2'
#     }
#
#     # 形状定义
#     class_markers = {
#         'mitotic_figure': 'o',
#         'not_mitotic_figure': 's'
#     }
#
#     # 获取所有组合类别名称
#     combined_class_names = le_combined.inverse_transform(range(len(le_combined.classes_)))
#
#     print(f"🔍 实际找到的组合类别: {combined_class_names}")
#
#     # 绘制3D散点图
#     legend_handles = []
#     legend_labels = []
#
#     for i, combined_name in enumerate(combined_class_names):
#         # 解析组合名称
#         actual_tissue = None
#         cell_type = None
#
#         for tissue in tissue_types:
#             if tissue in combined_name:
#                 actual_tissue = tissue
#                 remaining = combined_name.replace(tissue + '_', '')
#                 if remaining in class_names:
#                     cell_type = remaining
#                 break
#
#         mask = [label == combined_name for label in valid_combined_labels]
#
#         if any(mask):
#             color = tissue_colors.get(actual_tissue, 'gray')
#             marker = class_markers.get(cell_type, 'o')
#
#             # 3D散点图
#             scatter = ax.scatter(
#                 features_lda_3d[mask, 0],
#                 features_lda_3d[mask, 1],
#                 features_lda_3d[mask, 2],
#                 c=[color],
#                 marker=marker,
#                 s=60,
#                 alpha=0.7,
#                 edgecolors='white',
#                 linewidth=0.5,
#                 depthshade=True  # 启用深度阴影
#             )
#
#             # 创建图例句柄
#             legend_scatter = plt.Line2D([0], [0], marker=marker, color='w',
#                                       markerfacecolor=color, markersize=8,
#                                       linestyle='None')
#             legend_handles.append(legend_scatter)
#             legend_labels.append(combined_name)
#
#
#     # 添加网格
#     ax.grid(True, alpha=0.3)
#
#     # 设置视角
#     ax.view_init(elev=20, azim=45)  # 调整视角
#
#     # 处理图例标签
#     re_legend_labels = []
#     for label in legend_labels:
#         label = label.replace("_not_mitotic_figure", " Non-mitotic").replace("_mitotic_figure", " Mitotic")
#         label = label.replace('human_melanoma', 'H7').replace('canine_soft_tissue_sarcoma', 'H6').replace('human_neuroendocrine_tumor', 'H5').replace('canine_cutaneous_mast_cell_tumor', 'H4').replace('canine_lymphosarcoma', 'H3').replace('human_breast_cancer', 'H1').replace('canine_lung_cancer', 'H2')
#         re_legend_labels.append(label)
#
#     # 排序图例
#     label_handle_pairs = list(zip(re_legend_labels, legend_handles))
#     label_handle_pairs_sorted = sorted(label_handle_pairs, key=lambda x: x[0])
#     re_legend_labels_sorted = [pair[0] for pair in label_handle_pairs_sorted]
#     legend_handles_sorted = [pair[1] for pair in label_handle_pairs_sorted]
#
#     # 添加图例
#     ax.legend(legend_handles_sorted, re_legend_labels_sorted,
#               loc='upper left',
#               bbox_to_anchor=(1.02, 0.7),
#               fontsize=18,
#               frameon=True,
#               framealpha=0.9,
#               title_fontsize=18,
#               markerscale=1.5)
#
#     # 调整布局
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.75)  # 减小右侧绘图区域占比，给图例留位置
#
#     # 保存图片
#     plt.savefig('3D_LDA_14_categories.png', dpi=300, bbox_inches='tight')
#     plt.show()
#
#     # 打印统计信息
#     print("\n📊 3D LDA分析完成!")
#     print(f"🎯 总组合类别数: {len(combined_class_names)}")
#     print(f"📏 数据点总数: {len(features_lda_3d)}")
#     print(f"📈 3D LDA累计解释方差: {lda_3d.explained_variance_ratio_.sum():.2%}")
#
#
# if __name__ == "__main__":
#     lda_14_categories_visualization()
#########################################################

# conf ={"num_clients":30,"num_classes":6,"Size of samples for labels in clients":[[[0,95],[1,53],[2,49],[3,47],[4,53],[5,50]],[[0,59],[1,48],[2,47],[3,46],[4,54],[5,48]],[[0,58],[1,59],[2,49],[3,52],[4,61],[5,62]],[[0,60],[1,52],[2,45],[3,50],[4,56],[5,54]],[[0,56],[1,47],[2,47],[3,44],[4,56],[5,52]],[[0,57],[1,51],[2,48],[3,55],[4,57],[5,57]],[[0,57],[1,51],[2,47],[3,48],[4,53],[5,52]],[[0,48],[1,41],[2,38],[3,46],[4,54],[5,54]],[[0,52],[1,49],[2,42],[3,50],[4,45],[5,50]],[[0,53],[1,47],[2,38],[3,54],[4,44],[5,58]],[[0,59],[1,54],[2,46],[3,53],[4,47],[5,57]],[[0,50],[1,52],[2,46],[3,51],[4,61],[5,60]],[[0,57],[1,55],[2,47],[3,49],[4,57],[5,62]],[[0,59],[1,54],[2,45],[3,54],[4,60],[5,51]],[[0,54],[1,48],[2,42],[3,59],[4,53],[5,72]],[[0,51],[1,51],[2,47],[3,69],[4,78],[5,70]],[[0,61],[1,48],[2,46],[3,64],[4,78],[5,71]],[[0,56],[1,58],[2,55],[3,57],[4,73],[5,65]],[[0,52],[1,40],[2,39],[3,73],[4,73],[5,83]],[[0,51],[1,51],[2,45],[3,66],[4,73],[5,68]],[[0,52],[1,47],[2,45],[3,85],[4,89],[5,90]],[[0,46],[1,42],[2,36],[3,62],[4,63],[5,72]],[[0,59],[1,51],[2,54],[3,68],[4,68],[5,72]],[[0,58],[1,59],[2,55],[3,68],[4,69],[5,72]],[[0,74],[1,65],[2,58],[3,65],[4,74],[5,73]],[[0,59],[1,55],[2,50],[3,78],[4,74],[5,76]],[[0,57],[1,51],[2,44],[3,70],[4,80],[5,74]],[[0,54],[1,51],[2,46],[3,72],[4,79],[5,80]],[[0,53],[1,49],[2,48],[3,60],[4,65],[5,69]],[[0,65],[1,65],[2,62],[3,62],[4,59],[5,70]]]}
# data = conf["Size of samples for labels in clients"]
# clients = conf["num_clients"]
# categories = sorted({c for cli in data for c, _ in cli})
# cv = {c: i for i, c in enumerate(categories)}
# matrix = np.zeros((len(categories), clients))
#
# for cli_idx, cli in enumerate(data[:clients]):
#     for c, v in cli:
#         matrix[cv[c], cli_idx] = v
#
# # 绘制热力图
# plt.figure(figsize=(10, 8))
# # base = plt.cm.YlGnBu(np.linspace(0.1, 0.75, 255))   # 变浅
# # # 把 0 对应位置替换成原色图最浅颜色
# # origin_light = plt.cm.YlGnBu(0.0)                  # 原最浅颜色
# # colors = np.vstack([origin_light, base])           # 0 索引放原浅色
# # light_cmap = LinearSegmentedColormap.from_list('light_YlGnBu', colors, N=256)
#
# # 变浅
# plt.imshow(matrix, aspect='auto',cmap='YlGnBu', interpolation='nearest')
# sparse_xticks = range(1, clients+1, 2)
# plt.xticks([x-1 for x in sparse_xticks], sparse_xticks)
# plt.colorbar()
# ax = plt.gca()
# ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
# plt.tight_layout()
# plt.show()

#############################################
import matplotlib.pyplot as plt
import numpy as np

n = 20
values = np.full(n, 1000)
# 4 种颜色，每 4 根柱子一组，正好 5 组
colors = np.repeat(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 4)
plt.figure(figsize=(10, 8))
plt.bar(range(1, n + 1), values, color=colors, width=0.7)

plt.xticks(range(1, n + 1))
plt.ylim(0, 1100)
plt.tight_layout()
plt.show()
