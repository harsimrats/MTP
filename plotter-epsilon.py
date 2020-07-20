import pickle as pkl
import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np
# % matplotlib
# inline
if __name__=="__main__":

    ssTableData = {}
    separator = ":"
    # namePrefix = sys.argv[3]
    # namePrefix = "svmsimplebagofword5"
    namePrefix = "epsilonbagofword4lessdata"
    namePrefixOri = namePrefix

    takesample = False

    # typedata = sys.argv[1]
    typedata = "vdisk"
    if typedata == "vdisk":
        import vdiskHelper as dataHelper
    # typemodel = sys.argv[2]
    # typemodel = "svmsimple"
    typemodel = "epsilon"
    if typemodel == "epsilon":
        import epsilonHelper as modelHelper
    elif typemodel == "svmsimple":
        import svmsklearncoreHelper as modelHelper
    elif typemodel == "libsvmsch":
        import svmlibsvmHelper as modelHelper
    elif typemodel == "isolationforest":
        print("isolationforest loaded")
        import isolationForestHelper as modelHelper

    f = open("Observations/compiledobs_" + namePrefix + ".pkl", "rb")
    wholejson = pkl.load(f)
    f.close()
    fprs = []
    sizes = []
    bfsizes = []
    nsv = []
    nus = []
    gammas = []
    tols = []
    Ds = []
    epsilons = []
    agg_thrs = []
    box_thrs = []

    for i in wholejson["data"]:
        #     if i["fpr"] >0.31:
        #         continue
        #     if i["nsv"]*1.0/839364 > 0.17:
        #         continue
        fprs.append(i["fpr"])
        sizes.append(i["size"])
        bfsizes.append(i["bfsize"])
        #     nsv.append(i["nsv"]*1.0/839364)
        #     nus.append(i["nu"])
        #     gammas.append(i["gamma"])
        #     tols.append(i["tol"])
        Ds.append(i["D"])
        epsilons.append(i["epsilon"])
        agg_thrs.append(i["agg_thr"])
        box_thrs.append(i["box_thr"])
    #     print(i)
    # print(x, y)
    # print(sizes)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    # plt.scatter(fprs,nsv, color='green', alpha=0.7)
    # for i in range(len(fprs)):
    #     plt.annotate("nu:"+str(nus[i])+" gamma: "+str(gammas[i])+" tol: "+str(tols[i]), xy=(fprs[i],nsv[i]))
    # for i in range(len(fprs)):
    #     plt.annotate("D:"+str(Ds[i])+" epsilon: "+str(epsilons[i])+" agg_thr: "+str(agg_thrs[i])+" box_thr: "+str(box_thrs[i]), xy=(fprs[i],bfsizes[i]))

    print("fprsConsize made")
    indices = [(x, fprs[x], sizes[x]) for x in range(len(fprs))]
    from operator import itemgetter, attrgetter

    indices = sorted(indices, key=itemgetter(1, 2))
    print(indices)
    textIndices = [x[0] for x in indices[:50]]
    print("textIndices made", textIndices)
    # for i in fprsConsize:
    #     indices = fprsConsize[i]

    # texts = [
    #     plt.text(fprs[i], sizes[i], "D:"+str(Ds[i])+" epsilon: "+str(epsilons[i])+" agg_thr: "+str(agg_thrs[i])+" box_thr: "+str(box_thrs[i])
    #              ) for i in range(len(fprs))]
    # texts = [
    #     plt.text(fprs[i], sizes[i]
    #              # , "D:"+str(Ds[i])+" epsilon: "+str(epsilons[i])+" agg_thr: "+str(agg_thrs[i])+" box_thr: "+str(box_thrs[i])
    #              , "D: %.4f epsilon: %.4f agg_thr: %.4f box_thr: %.4f" % (Ds[i],epsilons[i], agg_thrs[i], box_thrs[i])
    #              ,rotation = 90
    #             , ha = 'right', va = 'bottom'
    #              ) for i in textIndices]
    # texts = [
    #     plt.text(fprs[i],sizes[i], str(epsilons[i]), ha='center', va='center') for i in range(len(fprs))]

    # iterations = adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black')
    #                          , expand_text=(2, 2)
    #                          , precision=0.001
    #                          , only_move={'points': 'y', 'text': 'y'}
    #                          # , lim=1
    #                          )
    ax1.set_ylabel("sizes")
    # ax2 = ax1.twinx()
    y = np.linspace(min(sizes), max(sizes), 500)
    f = open("Observations/compiledobs_" + "alwaysNegativebagofword4lessdata" + ".pkl", "rb")
    alwaysNegativeJson = pkl.load(f)
    f.close()
    c = bfsizes
    # plt.scatter(fprs, sizes, color='blue', alpha=0.7)
    plt.scatter(fprs, sizes, c=c, alpha=0.7)
    plt.colorbar(label="Size of bloom filters in bytes")

    plt.axvline(x=alwaysNegativeJson["fpr"], linestyle="dashed", color="green", label="Core bf fpr: %.3f"%(alwaysNegativeJson["fpr"]))
    plt.axhline(y=alwaysNegativeJson["bfsize"], linestyle="dashed", color="red", label="Core bf size: %d"%(alwaysNegativeJson["bfsize"]))
    # for i in range(len(fprs)):
    #     plt.plot([fprs[i], fprs[i]], [nsv[i],bfsizes[i]], 'ro-')
    # ax2.set_ylabel("bloom filter sizes (blue)")
    # plt.scatter(fprs,nsv)
    ax1.set_xlabel("false positive rate")
    plt.ylabel("Size of model in bytes")
    # plt.vlines(x=[0.3], ymin=min(y), ymax=max(y), color ='red')
    plt.legend()
    # texts = [
    #     plt.text(fprs[i], sizes[i]
    #              , "D: %.4f epsilon: %.4f agg_thr: %.4f box_thr: %.4f" % (Ds[i],epsilons[i], agg_thrs[i], box_thrs[i])
    #              ) for i in textIndices]
    # texts = [
    #     plt.text(fprs[i], sizes[i], "%d" % (sizes[i]), ha='center', va='center') for i in textIndices]

    # iterations = adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black')
    #                          , expand_text=(2, 2)
    #                          , precision=0.001
    #                          # , only_move={'points':'y', 'text':'y'}
    #                          # ,lim=5
    #                          # ,text_from_points=False
    #                          # ,text_from_text=False
    #                          # ,avoid_self=False
    #                          )
    # print("iterations to adjust ", iterations)

    plt.title('Epsilon OCC, bag of words (4) sparse vdisk data')
    plt.show()
    weights = [1 / ((sizes[x]) * (fprs[x])) for x in range(len(fprs))]
    weightedD = sum([Ds[x] * weights[x] for x in range(len(fprs))]) / sum(weights)
    weightedEpsilon = sum([epsilons[x] * weights[x] for x in range(len(fprs))]) / sum(weights)
    weightedAgg_thr = sum([agg_thrs[x] * weights[x] for x in range(len(fprs))]) / sum(weights)
    weightedBox_thr = sum([box_thrs[x] * weights[x] for x in range(len(fprs))]) / sum(weights)

    print("weighted d: %.3f, epsilon: %.3f, agg_thr: %.3f, box_thr: %.3f" % (weightedD, weightedEpsilon, weightedAgg_thr, weightedBox_thr))
    print("least fpr: %.3f" %(np.min(fprs)))