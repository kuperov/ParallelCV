import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def plot_model_results(results, title):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    ((p_diff, p_rhat), (p_ess, p_err)) = axes
    K = results['num_folds']
    draws, ess = results['fold_draws'] * 1e-3 * K, results['model_ess'] * 1e-3 * K
    diff_elpd, diff_se = results['diff_elpd'], results['diff_se']
    diff_mcse, diff_cvse = results['diff_mcse'], results['diff_cvse']
    tcrit = tfd.StudentT(df=results['num_folds']+2, loc=0, scale=1.).quantile(0.975)

    # mean
    line_m = p_diff.plot(draws, diff_elpd, linestyle='solid')
    # standard error
    line_se = p_diff.plot(draws, diff_elpd + tcrit*diff_se, linestyle='dashed')
    p_diff.plot(draws, diff_elpd - tcrit*diff_se, linestyle='dashed', color=line_se[0].get_color())
    # Monte Carlo standard error
    line_mcse = p_diff.plot(draws, diff_elpd + 1.96*diff_mcse, linestyle='dotted')
    p_diff.plot(draws, diff_elpd - 1.96*diff_mcse, linestyle='dotted', color=line_mcse[0].get_color())
    p_diff.axhline(y=0, linestyle='solid', linewidth=0.5)
    p_diff.set_title(r'Model $\widehat{elpd}_{CV}$ difference')
    p_diff.set_xlabel('Num. draws (all folds, chains)')
    p_diff.set_ylabel(r'$\widehat{elpd}_{CV}$ difference')
    p_diff.legend([line_m[0], line_se[0], line_mcse[0]], ['Estimate', 'Total error', 'Monte Carlo error'])

    model_rhat = results['model_max_rhat']
    plot_handles = []
    for m in [0, 1]:
        line = p_rhat.plot(draws, model_rhat[:, m], linestyle='solid', label=f'Model {["A","B"][m]}')
        plot_handles.append(line[0])
    p_rhat.axhline(1., linestyle='dashed', linewidth=0.5)
    p_rhat.set_title(r'Model fold max $\widehat{R}$')
    p_rhat.legend(handles=plot_handles)
    p_rhat.set_ylabel(r'Model $\widehat{R}$')
    p_rhat.set_xlim(left=0)
    p_rhat.set_ylim(bottom=min(1, float(jnp.min(model_rhat))), top=min(100, float(jnp.max(model_rhat))))

    p_ess.plot(draws, ess, linestyle='solid')
    p_ess.set_title(r'Model $\widehat{ESS}$ by draw')
    p_ess.legend(['Model A', 'Model B'])
    p_ess.set_ylabel(r"$\widehat{ESS}$ per model ('000)")

    p_err.plot(draws, diff_cvse, label='Cross-validation SE', linestyle='dashed')
    p_err.plot(draws, diff_mcse, label='Monte Carlo SE', linestyle='dotted')
    p_err.legend()
    p_err.set_title(r'$\widehat{elpd}_{CV}$ difference error components')
    p_err.set_ylabel('Standard error (nats)')

    if jnp.sum(results['stop'])>0:
        stop_drawk = draws[results['stop']][0]
        stopline = p_diff.axvline(x=stop_drawk, linestyle='dashed', linewidth=0.5)
        p_rhat.axvline(x=stop_drawk, color=stopline.get_color(), linestyle='dashed', linewidth=0.5)
        p_err.axvline(x=stop_drawk, color=stopline.get_color(), linestyle='dashed', linewidth=0.5)
        p_ess.axvline(x=stop_drawk, color=stopline.get_color(), linestyle='dashed', linewidth=0.5)

    for rax in axes:
        for ax in rax:
            ax.set_xlabel("Draws ('000, all folds and chains)")

    fig.suptitle(title)
    fig.tight_layout()


def plot_fold_results(results, title, show_legend=True):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    (p_ess, p_rhat), (p_elpds, p_divs) = axes
    drawsk, essk = results['fold_draws'] * 1e-3, results['fold_ess'] * 1e-3
    K = results['num_folds']
    tcrit = tfd.StudentT(df=results['num_folds']+2, loc=0, scale=1.).quantile(0.975)

    p_ess.plot(drawsk, essk[:,:K], linestyle='solid')
    p_ess.plot(drawsk, essk[:,K:], linestyle='dashed')
    p_ess.set_title(r'$\widehat{ESS}$')
    p_ess.set_ylabel(r"$\widehat{ESS}$ per fold ('000)")

    p_rhat.plot(drawsk, results['fold_rhat'][:,:K], linestyle='solid')
    p_rhat.plot(drawsk, results['fold_rhat'][:,K:], linestyle='dashed')
    p_rhat.set_title(r'$\widehat{R}$')
    p_rhat.set_ylabel(r'Per-fold $\widehat{R}$')
    if show_legend:
        p_rhat.legend([f'model {"A" if i < K else "B"} fold {i % K}' for i in range(2*K)], ncol=2)
    p_rhat.set_ylim(bottom=1., top=min(100, jnp.nanmax(results['fold_rhat'])))

    fold_diff, fold_mcse = results['fold_elpd_diff'], results['fold_mcse']
    handles = []
    for k in range(K):
        line = p_elpds.plot(drawsk, fold_diff[:,k], linestyle='solid', label=f'fold {k}')
        p_elpds.plot(drawsk, fold_diff[:,k] + tcrit * fold_mcse[:,k], linestyle='dotted', color=line[0].get_color())
        p_elpds.plot(drawsk, fold_diff[:,k] - tcrit * fold_mcse[:,k], linestyle='dotted', color=line[0].get_color())
        handles.append(line[0])
    p_elpds.axhline(y=0, linestyle='solid', linewidth=0.5)
    p_elpds.set_title(r'$\widehat{elpd}_{CV}$ differences')
    p_elpds.set_ylabel(r'Per-fold $\widehat{elpd}_{CV}$ differences')
    if show_legend:
        p_elpds.legend(handles=handles, ncol=2)

    divs = results['fold_divergences']
    for k in range(K):
        p_divs.plot(drawsk, divs[:,k], linestyle='solid')
    p_divs.set_title('Divergences')

    for rax in axes:
        for ax in rax:
            ax.set_xlabel("Draws per fold ('000; total of all chains)")

    fig.suptitle(title)

    plt.tight_layout()
    plt.show()
