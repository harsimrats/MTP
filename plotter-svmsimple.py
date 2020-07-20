import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
# % matplotlib
# inline
if __name__=="__main__":
    ssTableData = {}
    separator = ":"
    # namePrefix = sys.argv[3]
    namePrefix = "svmsimplebagofword4lessdata"
    # namePrefix = "epsilonbagofword8"
    namePrefixOri = namePrefix

    takesample = False

    # typedata = sys.argv[1]
    typedata = "vdisk"
    if typedata == "vdisk":
        import vdiskHelper as dataHelper
    # typemodel = sys.argv[2]
    typemodel = "svmsimple"
    # typemodel = "epsilon"
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
    totalDataPoints = None
    for i in wholejson["data"]:
        #     if i["fpr"] >0.31:
        #         continue
        #     if i["nsv"]*1.0/839364 > 0.17:
        #         continue
        fprs.append(i["fpr"])
        sizes.append(i["size"])
        bfsizes.append(i["bfsize"])
        if totalDataPoints is None:
            totalDataPoints = i["totalDataPoints"]
        nsv.append(i["nsv"] * 1.0 / totalDataPoints)
        nus.append(i["nu"])
        gammas.append(i["gamma"])
        tols.append(i["tol"])
        print(i)
    # print(x, y)
    # print(sizes)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    # plt.scatter(fprs,nsv, color='green', alpha=0.7)
    # for i in range(len(fprs)):
    #     plt.annotate("nu:"+str(nus[i])+" gamma: "+str(gammas[i])+" tol: "+str(tols[i]), xy=(fprs[i],nsv[i]))
    # for i in range(len(fprs)):
    #     plt.annotate(str(nus[i])+", "+str(gammas[i])+", "+str(tols[i]), xy=(fprs[i],nsv[i]))
    # texts = [plt.text(fprs[i],nsv[i], str(nus[i])+", "+str(gammas[i])+", "+str(tols[i]), ha='center', va='center') for i in range(len(fprs))]
    # texts = [
    #     plt.text(fprs[i], sizes[i], "nu:" + str(nus[i]) + " gamma: " + str(gammas[i]) + " tol: " + str(tols[i])
    #              ) for i in range(len(fprs))]
    fprsConsize = dict()
    for i in range(len(fprs)):
        fpr = fprs[i]
        if fpr in fprsConsize:
            fprsConsize[fpr].append(i)
        else:
            fprsConsize[fpr] = [i]
    fprsSizeConsize = dict()
    for i in range(len(fprs)):
        fpr = fprs[i]
        size = sizes[i]
        if (fpr,size) in fprsSizeConsize:
            fprsSizeConsize[(fpr,size)].append(i)
        else:
            fprsSizeConsize[(fpr,size)] = [i]
    print("fprsConsize made")
    indices = [(x, fprs[x], sizes[x]) for x in range(len(fprs))]
    from operator import itemgetter, attrgetter
    indices = sorted(indices, key=itemgetter(1,2))
    print(indices)

    textIndices = [x[0] for x in indices[:20]]
    # for i in fprsConsize:
    #     minsize = min([sizes[j] for j in fprsConsize[i]])
    #     textIndices.append(fprsSizeConsize[(i,minsize)][0])
    print("textIndices made", textIndices)


    # ax1.set_ylabel("support vector ratio (green)")
    # ax2 = ax1.twinx()


    y = np.linspace(min(sizes), max(sizes), 500)
    f = open("Observations/compiledobs_" + "alwaysNegativebagofword4lessdata" + ".pkl", "rb")
    alwaysNegativeJson = pkl.load(f)
    f.close()
    # line1, = plt.plot(alwaysNegativeJson["fpr"], , '--', linewidth=2,
    #                  label='Dashes set retroactively')

    # plt.scatter(fprs, sizes, color='blue', alpha=0.7)
    c=bfsizes
    print(np.min(c),np.max(c))
    plt.scatter(fprs, sizes,  alpha=0.7, c=c)
    plt.set_cmap('viridis')
    plt.colorbar(label="Size of bloom filters in bytes")

    plt.axvline(x=alwaysNegativeJson["fpr"], linestyle="dashed", color="green", label="Core bf fpr: %.3f"%(alwaysNegativeJson["fpr"]))
    plt.axhline(y=alwaysNegativeJson["bfsize"], linestyle="dashed", color="red", label="Core bf size: %d"%(alwaysNegativeJson["bfsize"]))
    # for i in range(len(fprs)):
    #     plt.plot([fprs[i], fprs[i]], [nsv[i],bfsizes[i]], 'ro-')
    # ax2.set_ylabel("bloom filter sizes (blue)")
    # plt.scatter(fprs,nsv)
    ax1.set_xlabel("false positive rate")
    plt.ylabel("size of models in bytes")
    # plt.vlines(x=[0.3], ymin=min(y), ymax=max(y), color ='red')
    # texts = [
    #     plt.text(fprs[i],sizes[i],"%d, %d" %(sizes[i], bfsizes[i]), ha='center', va='center') for i in textIndices]
    # texts = [
    #     plt.text(fprs[i], sizes[i], "nu: %.3f gamma: %.3f tol: %.3f" %(nus[i],gammas[i],tols[i])
    #              ) for i in textIndices]
    # texts = [
    #     plt.text(fprs[i], sizes[i], "%d" % (bfsizes[i]), ha='center', va='center') for i in range(len(fprs))]
    #
    #
    # iterations = adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black')
    #                          , expand_text=(2,2)
    #                          , precision=0.001
    #                          # , only_move={'points':'y', 'text':'y'}
    #                          # ,lim=5
    #                          # ,text_from_points=False
    #                          # ,text_from_text=False
    #                          # ,avoid_self=False
    #                          )
    # print("iterations to adjust ", iterations)
    plt.title('One Class svm sklearn, bag of words (4) sparse vdisk data')
    plt.legend()
    plt.show()
    weights = [1/((sizes[x])*(fprs[x])) for x in range(len(fprs))]
    weightedNu = sum([nus[x]*weights[x] for x in range(len(fprs))])/sum(weights)
    weightedGamma = sum([gammas[x] * weights[x] for x in range(len(fprs))]) / sum(weights)
    weightedTol = sum([tols[x] * weights[x] for x in range(len(fprs))]) / sum(weights)
    print("weighted nu: %.3f, gamma: %.3f, tol: %.3f" %(weightedNu,weightedGamma,weightedTol))

