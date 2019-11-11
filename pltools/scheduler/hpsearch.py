import nevergrad as ng

#TODO: typing


def get_instrumentation_from_config(cfg):
    var_type = ng.instrumentation.utils.Variable
    var_keys = cfg.nested_get_key_fn(lambda x: isinstance(x, var_type))
    search_vars = {key: cfg.get_with_dot_str(key) for key in var_keys}
    return ng.Instrumentation(**search_vars)


def update_config_with_dot_str(dict_like, params):
    for key, item in params.items():
        dict_like.set_with_dot_str(key, item)
    return dict_like


def hyperparameter_search(cfg, single_run, optim_cls, budget, optim_kwargs=None, **kwargs):
    optim_kwargs = {} if optim_kwargs is None else optim_kwargs

    instrumentation = get_instrumentation_from_config(cfg)
    optimizer = optim_cls(instrumentation=instrumentation, budget=budget, **optim_kwargs)

    for u in range(budget):
        params = optimizer.ask()
        cfg = update_config_with_dot_str(cfg, params.kwargs)
        result = single_run(cfg, **kwargs)
        optimizer.tell(params, result)
    return optimizer.recommend()
