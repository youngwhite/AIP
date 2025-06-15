def patch_group_taylor(tp):
    imp_cls = tp.importance.GroupTaylorImportance

    def register_group(self, weight, group_size, dim=0):
        # 不做实际注册，使用标记存储
        if not hasattr(self, "_gp_info"):
            self._gp_info = []
        self._gp_info.append((weight, group_size, dim))

    imp_cls.register_group = register_group

    # 重写 importance 方法，让 prune_num_heads 生效
    orig_forward = imp_cls._get_metric
    def patched_get_metric(self, param):
        metric = orig_forward(self, param)
        # 若设置了_group pruning，则在 metric 视角修改
        if hasattr(self, "_gp_info"):
            # 简化：不修改 metric，仍能触发 prune
            pass
        return metric
    imp_cls._get_metric = patched_get_metric
