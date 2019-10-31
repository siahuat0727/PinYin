import matplotlib.pyplot as plt


# x = ["no clipping-@ magic" , "head clipping-@ magic" , "tail clipping-@ magic" , "both clipping-@ magic "]
# y = [0.8665210475082504 , 0.8699399320971533, 0.870201096892139 , 0.8735250124646834]
# plt.xlabel('pattern of clipping-@ magic')
# plt.ylabel('accuracy')
# plt.ylim([0.865, 0.875])
# x_pos = [i for i in range(len(x))]
# plt.bar(x_pos, y, width=0.4)
# plt.xticks(x_pos, x)
# plt.title('Plot of accuracy against the pattern of clipping-@ magic using 2-gram model without smoothing')

# x = [0.8, 0.85, 0.875, 0.90, 0.925, 0.95, 1.0]
# y = [0.8742016667062371, 0.8758161399843302, 0.876302856193167, 0.8768014435290487, 0.8760891759063606, 0.8760060780170469, 0.8735250124646834]
# plt.xlabel('lambda')
# plt.ylabel('accuracy')
# plt.title('Plot of accuracy against lambda (smoothing factor) using 2-gram model with both clipping-@ magic')
# plt.plot(x, y, '*-')

# x = ["2-gram", "3-gram", "4-gram"]
# y = [0.8735250124646834, 0.9314561124433154, 0.9403238443457822]
# plt.xlabel('n-gram model')
# plt.ylabel('accuracy')
# plt.ylim([0.85, 0.95])
# x_pos = [i for i in range(len(x))]
# plt.bar(x_pos, y, width=0.3)
# plt.xticks(x_pos, x)
# plt.title('Plot of accuracy against different n-gram model with both clipping-@ magic and lambda = 0.9')

x = ["3-gram fast", "3-gram perfect", "4-gram fast", "4-gram perfect + slim"]
y = [0.9314561124433154, 0.9379971034450011, 0.9403238443457822, 0.9481944015764857]
plt.xlabel('search method (fast v.s. perfect)')
plt.ylabel('accuracy')
plt.ylim([0.925, 0.95])
x_pos = [0, 1, 2.5, 3.5]
plt.bar(x_pos, y, width=0.3)
plt.xticks(x_pos, x)
plt.title('Plot of accuracy against different search method (model with both clipping-@ magic and lambda = 0.9)')

plt.show()
