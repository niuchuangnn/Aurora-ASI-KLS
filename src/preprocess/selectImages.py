"""According to some self-define rules, this function selects the images for
following experiments."""

import src.util.paseLabeledFile as plf

filePath = '../../Data/Alllabel2003_38044.txt'
file_selected = '../../Data/one_in_minute.txt'
file_balance = '../../Data/balanceSampleFrom_one_in_minute.txt'
file_own = '../../Data/one_in_minute_selected.txt'
class_num = 4
select_num_perclass = 500

[names, labels] = plf.parseNL(filePath)

arrangedImgs = plf.arrangeToClasses(names, labels, class_num)

print 'total labeled number:'
for i in range(4):
    print 'NO. class ' + str(i+1), len(arrangedImgs[str(i+1)])

# print plf.timeDiff(names[0], names[2])
[ids, sampledImgs] = plf.sampleImages(names)

f_w = open(file_selected, 'w+')

for i in range(len(ids)):
    f_w.write(sampledImgs[i] + ' ' + labels[ids[i]] + '\n')

f_w.close()

[names_s, labels_s] = plf.parseNL(file_selected)
arrangedImgs_s = plf.arrangeToClasses(names_s, labels_s, class_num, [['1'], ['2'], ['3'], ['4']])

print 'one in minute number:'
for i in range(class_num):
    print 'NO. class ' + str(i+1), len(arrangedImgs_s[str(i+1)])

balanceImgs = plf.balanceSample(arrangedImgs_s, select_num_perclass)
f_b = open(file_balance, 'w+')

print 'balance selected number:'
for c in balanceImgs:
    print 'NO. class ' + c, len(balanceImgs[c])
    balanceImgs[c].sort()
    for file in balanceImgs[c]:
        f_b.write(file + ' ' + str(c) + '\n')
f_b.close()

print 'test sampled labeled file:'
print plf.compareLabeledFile(filePath, file_selected)
print plf.compareLabeledFile(filePath, file_balance)

type3_file = '../../Data/type3_1000_500_500.txt'
f2 = open(type3_file, 'w')

arrangedImgs_s3, rawTypes = plf.arrangeToClasses(names_s, labels_s, 3, [['1'], ['2'], ['3']])
print 'class1: ' + str(len(arrangedImgs_s3['1']))
print 'class2: ' + str(len(arrangedImgs_s3['2']))
print 'class3: ' + str(len(arrangedImgs_s3['3']))

balance3Imgs = plf.balanceSample(arrangedImgs_s3, 1000)
for c in balance3Imgs:
    print 'NO. class ' + c, len(balance3Imgs[c])

imgs_s23 = balance3Imgs.copy()
imgs_s23.pop('1')
imgs_s23 = plf.balanceSample(imgs_s23, 500)

imgs_s123 = {}
imgs_s123['1'] = balance3Imgs['1']
imgs_s123['2'] = imgs_s23['2']
imgs_s123['3'] = imgs_s23['3']

for c in imgs_s123:
    print 'NO. class ' + c, len(imgs_s123[c])
    imgs_s123[c].sort()
    for file in imgs_s123[c]:
        f2.write(file + ' ' + str(c) + '\n')
f2.close()
print plf.compareLabeledFile(filePath, type3_file)

# type4_file = '../../Data/type4_1500_500_500_500.txt'
# f4 = open(type4_file, 'w')
#
# arrangedImgs_s4 = plf.arrangeToClasses(names_s, labels_s, 4, [['1'], ['2'], ['3'], ['4']])
# print 'class1: ' + str(len(arrangedImgs_s4['1']))
# print 'class2: ' + str(len(arrangedImgs_s4['2']))
# print 'class3: ' + str(len(arrangedImgs_s4['3']))
# print 'class4: ' + str(len(arrangedImgs_s4['4']))
#
# # balance4Imgs = plf.balanceSample(arrangedImgs_s4, 1000)
# # for c in balance3Imgs:
# #     print 'NO. class ' + c, len(balance3Imgs[c])
# #
# imgs_s1 = arrangedImgs_s4.copy()
# imgs_s1.pop('2')
# imgs_s1.pop('3')
# imgs_s1.pop('4')
# imgs_s1 = plf.balanceSample(imgs_s1, 1500)
#
# imgs_s234 = arrangedImgs_s4.copy()
# imgs_s234.pop('1')
# imgs_s234 = plf.balanceSample(imgs_s234, 500)
#
# imgs_s1234 = {}
# imgs_s1234['1'] = imgs_s1['1']
# imgs_s1234['2'] = imgs_s234['2']
# imgs_s1234['3'] = imgs_s234['3']
# imgs_s1234['4'] = imgs_s234['4']
#
# for c in imgs_s1234:
#     print 'NO. class ' + c, len(imgs_s1234[c])
#     imgs_s1234[c].sort()
#     for file in imgs_s1234[c]:
#         f4.write(file + ' ' + str(c) + '\n')
# f4.close()
# print plf.compareLabeledFile(filePath, type4_file)
file_38044_selected = '../../Data/labeled2003_38044_G_selected.txt'
[names_own, labels_own] = plf.parseNL(file_38044_selected)
arrangedImgs_s = plf.arrangeToClasses(names_s, labels_s, class_num, [['1'], ['2'], ['3'], ['4']])

# type4_file_own = '../../Data/type4_600_300_300_300.txt'
# type4_file_own = '../../Data/type4_300_300_300_300.txt'
type4_file_own = '../../Data/type4_b500.txt'
f4 = open(type4_file_own, 'w')

arrangedImgs_own = plf.arrangeToClasses(names_own, labels_own, 4, [['1'], ['2'], ['3'], ['4']])
print 'class1: ' + str(len(arrangedImgs_own['1']))
print 'class2: ' + str(len(arrangedImgs_own['2']))
print 'class3: ' + str(len(arrangedImgs_own['3']))
print 'class4: ' + str(len(arrangedImgs_own['4']))

# balance4Imgs = plf.balanceSample(arrangedImgs_s4, 1000)
# for c in balance3Imgs:
#     print 'NO. class ' + c, len(balance3Imgs[c])
#
imgs_s1234 = plf.balanceSample(arrangedImgs_own, 500)

# imgs_s1 = arrangedImgs_own.copy()
# imgs_s1.pop('2')
# imgs_s1.pop('3')
# imgs_s1.pop('4')
# imgs_s1 = plf.balanceSample(imgs_s1, 600)
#
# imgs_s234 = arrangedImgs_own.copy()
# imgs_s234.pop('1')
# imgs_s234 = plf.balanceSample(imgs_s234, 300)
#
# imgs_s1234 = {}
# imgs_s1234['1'] = imgs_s1['1']
# imgs_s1234['2'] = imgs_s234['2']
# imgs_s1234['3'] = imgs_s234['3']
# imgs_s1234['4'] = imgs_s234['4']

for c in imgs_s1234:
    print 'NO. class ' + c, len(imgs_s1234[c])
    imgs_s1234[c].sort()
    for file in imgs_s1234[c]:
        f4.write(file + ' ' + str(c) + '\n')
f4.close()
print plf.compareLabeledFile(filePath, type4_file_own)