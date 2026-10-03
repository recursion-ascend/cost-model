"""测试用的 StageLink 快捷构造.

两条边的"硬件强制同核"属性 (gmm1->activation) 每次都要写一遍太吵, 收在这里。
"""
import moe_cost_model as m

#: gmm1->activation: Fixpipe 直给配对 AIV0, 同核是硬件强制; depth = UB 槽数
def hw(depth: int = 1):
    return m.StageLink("gmm1", "activation", location="onchip", depth=depth,
                       colocated_by_hardware=True)


def links(depth: int = 1, **act_to_gmm2):
    """(gmm1->act, act->gmm2) 两条边; act_to_gmm2 的字段直接透传."""
    return (hw(depth), m.StageLink("activation", "gmm2", **act_to_gmm2))
