# 人工调整打点

唯一阶段配置是 `include/profile_stages.h` 中非作用域枚举 `MoeProfileStage`，编译器识别真实枚举，IDE可补全。已有ID不要重排；新阶段增加一个未占用的整数BEGIN ID，END为下一整数，唯一例外是兼容旧数据的KERNEL（1→255）。生成器检查重复及交叠。

```cpp
void SomeStage(/* ... */) {
    MOE_PROFILE_BIND(params.tilingData->profBufGm); // buffer有效后绑定一次
    MOE_PROFILE_BEGIN(GMM1, payload);
    // 实际计算，包含内部卡内搬运
    // 复用已有完成事件，确认异步工作完成
    MOE_PROFILE_END(GMM1, payload);
}
```

宏参数是枚举值，不是字符串，也不是token拼接。BIND在当前代码块引入枚举名称并绑定buffer；无Stage::前缀，不引入全局using。不同函数/作用域使用自身有效buffer，绑定须位于buffer变量初始化之后。同一作用域绑定一次。

- 移动边界：移动BEGIN/END；payload保持一致，提前return与分支必须保证配对。
- 新增阶段：枚举头文件加一项，再插BEGIN/END。`build/generated/profile_events_generated.h`和`build/generated/pair_map.json`自动生成，不手改。
- CMake 配置及枚举文件变更触发自动生成。独立生成可用当前 Python 执行 `scripts/generate_profile_events.py --stages include/profile_stages.h --config configs/two_card.json5 --output <构建目录>/generated`（工作目录为 megamoe_profile）。解析历史数据使用该次归档的事件头文件和配对表，不用当前枚举重新解释历史 ID。
- 动态选择阶段可使用枚举变量，例如 `auto stage = isLocal ? DISPATCH_LOCAL : DISPATCH_XFER;`。
- 阶段名与ID自动化不等于边界自动化。上游等待单独WAIT，不能把异步提交当完成；宏不插同步、不改pipeline。
- 新增阶段默认可以配对和显示。本示范只导出事件时间与原始payload，不推导shape、FLOP或带宽。
- ENABLE_PROFILING控制记录，关闭时仍进行枚举类型检查，不执行payload。普通构建同时关闭测量专用FIX_S/V_S，但未以OFF进行性能基线采集。

改内核后运行 `python megamoe_profile/run.py --config ... --name ...`，重新编译并导出。事件枚举修改后由CMake自动生成事件头文件与配对表。
