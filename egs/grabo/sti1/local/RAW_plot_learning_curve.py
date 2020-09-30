# Plot learning curves SVM
if False:
    for i, tn in enumerate(target_names):
        X = features[:, 0, :].numpy()
        y = target.values[:, i]
        plot_learning_curve(estimator, tn, X, y, n_jobs=-1)
        plt.savefig(f"exp/figures/learning_curve_text_fixed/feats_first/learning_curve_{tn}.png")
        plt.close('all')


    scores = pd.read_csv("exp/figures/learning_curve_text/feats_first/scores.csv", header=[0,1,2], index_col=0)                                                                                                     
    scores_fixed = pd.read_csv("exp/figures/learning_curve_text_fixed/feats_first/scores.csv", header=[0,1,2], index_col=0)                                                                                         
    averages = scores.xs("mean", axis=1, level=1).mean().unstack(level=0)
    stdevs = ((scores.xs("std", axis=1, level=1) ** 2).mean() ** (1/2)).unstack(level=0)
    train_sizes=np.linspace(.1, 1., 5)
    plt.fill_between(train_sizes, averages - stdevs, averages + stdevs, alpha=.1)
    scores = {fn: pd.read_csv(fn, header=[0,1,2], index_col=0) for fn in Path("exp/figures").glob("learning_curve_text*/scores.csv")}                                                                               
    scores = {fn: pd.read_csv(fn, header=[0,1,2], index_col=0) for fn in Path("exp/figures").glob("**/scores.csv")}                                                                                                 
    scores = {fn: scores.xs(("train", "mean"), axis=1)}
    scores = {fn: df.xs(("train", "mean"), axis=1) for fn, df in scores.items()}
    scores = {fn: pd.read_csv(fn, header=[0,1,2], index_col=0) for fn in Path("exp/figures").glob("**/scores.csv")}                                                                                                 
    scores = {fn: df.xs(("test", "mean"), axis=1) for fn, df in scores.items()}
    average_scores = {fn: df.mean() for fn, df in scores.items()}
    average_scores = {fn: df.mean().rename(fn.parent.name.split("_")[1] + ("_fixed" if "_fixed" in fn.parents[1].name else "")) for fn, df in scores.items()}                                                       
    average_scores = pd.concat(scores.values())
    average_scores = pd.concat(average_scores.values())
    average_scores = {fn: df.mean().rename(fn.parent.name.split("_")[1] + ("_fixed" if "_fixed" in fn.parents[1].name else "")) for fn, df in scores.items()}                                                       
    average_scores = pd.concat(average_scores.values())
    average_scores = {fn: df.mean().rename(fn.parent.name.split("_")[1] + ("_fixed" if "_fixed" in fn.parents[1].name else "")) for fn, df in scores.items()}                                                       
    average_scores = pd.concat(average_scores.values(), axis=1)
    average_scores.plot()
    plt.savefig("exp/figures/learning_curve_agg.png")

