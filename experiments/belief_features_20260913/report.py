"""Write a reviewable Chinese report from completed, verified experiment results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.belief_features_20260913.analyze import read, SEEDS, SPLITS

ARM = {'baseline': '现有观测基线', 'temporal': '长期轨迹', 'relational': '空间与攻防关系'}
SPLIT = {'test': '冻结策略测试', 'ood_attack': '偏好攻击的随机行为', 'ood_random': '均匀随机行为'}
GROUP = {'all': '全部待猜棋子', 'moved': '曾移动', 'never_moved': '未移动',
         'opening_zero': '初始局面', 'opening': '0–32 步', 'middle': '33–256 步',
         'late': '257 步及以后', 'moved_after_128': '至少 128 步且曾移动'}


def interval(value, *, percent=False):
    factor = 100 if percent else 1
    point = value['delta_candidate_minus_reference'] * factor
    lo, hi = [x * factor for x in value['paired_seed_and_game_ci95']]
    return f'{point:+.4f} [{lo:+.4f}, {hi:+.4f}]'


def evidence(value):
    lo, hi = value['paired_seed_and_game_ci95']
    changes = value['paired_seed_deltas']
    if hi < 0:
        return '本轮区间完全低于零' + ('，三个种子均改善' if all(x < 0 for x in changes) else '，但种子间方向有差异')
    if lo > 0:
        return '本轮区间完全高于零' + ('，三个种子均变差' if all(x > 0 for x in changes) else '，但种子间方向有差异')
    return '本轮区间跨零，无法确认稳定改善'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def make_report(root, result):
    assert result['status'] == 'all_experiments_and_diagnostics_verified'
    primary, expanded = result['primary'], result['expanded']
    early_path=root/'analysis_early/resolution_check.json'
    early=read(early_path) if early_path.exists() else None
    sections = ['# BeliefNet 公开特征与可学习性实验报告',
        '实验日期：2026-09-13。两台 H20；参数 EMA 全程关闭。所有数值来自完整运行、回收到本地并通过 SHA-256 校验的结果。']

    policy_rows = []
    for backend, label in [('cpu', 'CPU / FP32 / 新 512 局'), ('gpu', 'GPU / BF16 / 新 2,048 局')]:
        path = root / f'analysis_policy/current_{backend}.json'
        if not path.exists():
            continue
        policy_result = read(path)
        assert policy_result['status'] == 'all_paired_policy_results_verified'
        for role, name in [('A', '当前采样策略 guarded_s501'), ('B', '原始策略 ckpt_001000，在 v4 输入下复测')]:
            run = policy_result['runs'][role]
            m = run['summary']['metrics']
            policy_rows.append([label, name, int(m['eval/wins']), int(m['eval/losses']),
                                int(m['eval/draws']), int(m['eval/ongoing']),
                                f"{m['eval/win_rate'] * 100:.4f}%",
                                f"{run['both_teams_won_pairs']}/{run['independent_setup_pairs']}"])
    if policy_rows:
        sections += ['## 优先目标：对均匀随机对手的实际胜率',
            table(['测试', '权重', '胜', '负', '和', '未完成', '胜率', '双方均胜的布阵对'], policy_rows),
            '以上独立复测均只使用原始行棋参数和当前公开规则观测，不接入学习型 BeliefNet 或 EMA。'
            '两份策略在同一后端使用完全相同的布阵、随机种子及换边协议，逐局布阵哈希已核对。'
            'CPU 与 GPU 使用不同的测试种子及随机数流，数值精度也不同；有限测试集全胜不代表任意新局面都必胜。',
            '旧观测 v3 的独立结果是 512/512 胜；旧训练最后的另一组监控结果是 127/128 胜。'
            '这些历史结果与当前 v4 复测分开保留，不能只挑选全胜记录。'
            '原始 ckpt_001000 的本轮测试明确保持权重不变、改用当前输入；没有改写旧检查点。',
            'Ataraxos 的训练顺序也是先得到最终行棋与布阵策略，再训练供搜索使用的 belief。'
            '本项目因此优先稳定策略，同时完成以下离线特征对照；延后接入 BeliefNet 不等于取消公开规则推理。'
            '[论文 §2.5](https://arxiv.org/html/2511.07312v1#S2.SS5)；'
            '本地阶段配置与候选初始化已验证，但准备工作本身不算新增训练结果。']

    opening = []
    for arm in ('temporal', 'relational'):
        a = primary['paired_comparisons'][arm]['test']['nll']
        b = expanded['paired_comparisons'][arm]['test']['nll']
        opening.append(f"**{ARM[arm]}：**1,024 个训练对局时，主测试 NLL 差值为 {interval(a)}，{evidence(a)}；"
                       f"5,120 个训练对局时为 {interval(b)}，{evidence(b)}。")
    volume = result['data_volume_comparisons']['baseline']['test']['nll']
    opening.append(f"**数据量复核：**基线从 1,024 个独立训练对局扩至 5,120 个后，主测试 NLL 改变量为 {interval(volume)}；{evidence(volume)}。这项复核是观察到过拟合后追加的探索性实验。")
    sections += ['## 当前证据支持什么', '\n\n'.join(opening),
        '以下差值均为候选减去参照，NLL/Brier 越低越好，准确率越高越好。方括号是同时按本轮训练种子与完整测试对局配对重采样得到的 95% 区间。只有三组主训练种子，区间用于描述本轮证据；分组和分布比较没有做多重检验校正。']
    if early:
        assert early['status']=='early_resolution_verified'
        selections=early['selected_steps']['baseline']['dense_steps']
        difference=early['dense_selection_minus_primary_selection']['baseline']['test']['nll']
        sections.append(f"**早期选模分辨率复核：**每 50 步验证时，三个基线分别选中第 {', '.join(map(str,selections))} 步；相对主实验选模的测试 NLL 变化为 {interval(difference)}。具体特征比较见后文，不能把重跑相同种子当作额外独立重复。")

    sections += ['## 实验覆盖与共同基线', table(
        ['阶段', 'A：长期轨迹与基线', 'B：关系特征与基线', '每单元更新数', '结果状态'],
        [['主实验', '6/6', '8/8（含补充种子 604）', '50,000', '完成并校验'],
         ['扩充数据复核', '6/6', '6/6', '5,000', '完成并校验'],
         ['保存模型重评分', '12/12', '14/14', '无参数更新', '完成并校验']]),
        f"三组主配对种子为 {', '.join(map(str, SEEDS))}，每组参数数目均为 {primary['parameters_per_arm']:,}。"
        '两类特征均增加 32 个通道，通过零初始化适配层输入同一核心网络；基线保留同一适配层并输入零值，因此初始预测相同。特征组的适配层有有效梯度，基线对应梯度为零。',
        table(['阶段', '同种子初始化、数据、步数、最优步骤、最优/末期参数及测试指标'],
              [[label, '三组均完全一致' if all(x['passed'] for x in report['cross_host_baseline_agreement'])
                else '存在差异：详见 analysis.json，不能声称完全一致']
               for label, report in [('主实验', primary), ('扩充数据复核', expanded)]]),
        '训练批量为 256，Adam 学习率 5e-5，梯度裁剪 0.5，dropout 0.1，BF16 前向、FP32 概率与损失。每 500 步在独立验证集比较 NLL，选择最优原始参数；测试集不参与优化和 checkpoint 选择。训练目标是公开规则候选集合内、仍有歧义的存活敌方棋子身份交叉熵。没有 belief PPO 项或在线策略联训。',
        '本轮在公开规则允许的类型集合内归一化概率，且只统计仍有歧义的存活敌棋；与此前联合训练路径直接对 12 类 logits 做 log-softmax 的原始交叉熵口径不同。不能把两个任务的损失数值直接横向比较。',
        '采样策略采用上一轮 guarded_s501 checkpoint 的原始 policy 参数，并在本轮全程冻结。所有 BeliefNet 均从零初始化，不复用该历史文件中的 belief 或 EMA 参数。特征对照中该固定策略只用于生成数据；另行对随机对手的复测见前文，这仍不能证明对强对手的代表性。来源、哈希与保留文件见 source_inputs/SOURCE_INPUTS.json。']

    for title, prefix, report in [('1,024 个独立训练对局：主实验', 'training', primary),
                                   ('5,120 个独立训练对局：探索性复核', 'scale', expanded)]:
        sections += ['## ' + title]
        rows = []
        for split in SPLITS:
            for arm in ARM:
                score = report['primary_means'][arm][split]
                rows.append([SPLIT[split], ARM[arm], f"{score['nll']:.6f}", f"{score['brier']:.6f}",
                             f"{score['accuracy'] * 100:.3f}%", f"{score['ece'] * 100:.3f}%",
                             f"{score['wrong_confident_fraction'] * 100:.3f}%"])
        sections.append(table(['测试分布', '输入', 'NLL', 'Brier', '准确率', 'ECE', '高置信错误占全部标签'], rows))
        sections.append('ECE 使用 10 个固定置信度分桶；高置信错误指预测错误且置信度至少 90%，分母为全部有效待猜标签。')
        rows = []
        for arm in ('temporal', 'relational'):
            for split in SPLITS:
                comparison = report['paired_comparisons'][arm][split]
                rows.append([ARM[arm], SPLIT[split], interval(comparison['nll']),
                             interval(comparison['brier']), interval(comparison['accuracy'], percent=True),
                             ', '.join(f'{x:+.5f}' for x in comparison['nll']['paired_seed_deltas'])])
        sections += [table(['特征', '分布', 'NLL 差及区间', 'Brier 差及区间', '准确率差及区间（百分点）', '各种子 NLL 差'], rows),
                     f'![{title}曲线与配对比较](../analysis_{prefix}/belief_feature_results.png)']

    sections += ['## 从训练拟合到新对局泛化',
        '训练损失下降只说明拟合能力。为判断模型是否能推广到新对局，另对保存参数进行完整训练集重评分，并与独立验证、测试结果对照。下表的训练成绩都来自确定性的评估模式，关闭 dropout；它们不同于训练日志中含 dropout 的小批量平均损失。']
    rows = []
    for prefix, label, report in [('training', '1,024 对局', primary), ('scale', '5,120 对局', expanded)]:
        for arm in ARM:
            fit = result['full_training_fit'][prefix][arm]
            selected = result['selection_window_check'][arm]['primary_best_steps' if prefix == 'training' else 'expanded_best_steps']
            final_val = sum(x['last_metrics']['validation']['nll'] for x in report['learning_trajectories'][arm]) / 3
            rows.append([label, ARM[arm], ', '.join(map(str, selected)),
                         f"{fit['selected_training']['nll']:.5f}",
                         f"{fit['final_training']['nll']:.5f}",
                         f"{fit['final_training']['accuracy'] * 100:.2f}%", f'{final_val:.5f}',
                         f"{report['primary_means'][arm]['test']['nll']:.5f}"])
    sections.append(table(['数据', '输入', '选中步骤', '选中参数训练 NLL', '末期参数训练 NLL',
                           '末期参数训练准确率', '末期验证 NLL', '选中参数测试 NLL'], rows))
    rows = []
    for arm in ARM:
        for split in SPLITS:
            comparison = result['data_volume_comparisons'][arm][split]
            rows.append([ARM[arm], SPLIT[split], interval(comparison['nll']),
                         interval(comparison['accuracy'], percent=True)])
    sections += [table(['输入', '测试分布', '扩充数据 NLL − 原数据 NLL', '准确率变化（百分点）'], rows),
        '![数据量效应](data_volume_results.png)',
        '原主实验更新上限为 50,000，扩充复核上限为 5,000。' +
        ('所有主实验选中模型均位于前 5,000 步，图中另列相同前 5,000 步的验证曲线。'
         if all(x['all_selected_primary_steps_at_most_5000'] for x in result['selection_window_check'].values())
         else '有主实验最优模型位于 5,000 步之后，数据量比较同时受到选择窗口不同的影响。') +
        '即使选中步骤落在共同区间，两阶段的数据曝光次数与候选选择范围仍不完全相同；不能把这项探索性比较当作严格等 epoch 或等总计算量结论。']

    sections += ['## 长期轨迹在哪些局面得到检验',
        '现有观测已经包含最近 32 步的起点／终点平面，以及若干累计移动和战斗计数。本轮增加的是 32 维公开轨迹摘要：更长窗口、位移距离、方向变化和访问记录等；没有测试读取完整历史序列的循环网络或历史 Transformer。',
        '主测试以开局和中局为主，后期标签很少。“曾移动”同时包括空走和成功吃子后的位移；全部原始数据分片已逐标签验证该公开分组与独立轨迹记录一致。同一对局的多个棋子、时间点和四个观察者视角始终作为一个重采样单位。']
    audit=read(root/'dataset_audit.json')['shards']
    rows=[]
    for split in SPLITS:
        feature=audit[split]['features']
        values=[feature['temporal']['nonzero_by_channel'][i] for i in (2,7,9,21)]
        values.append(feature['relational']['nonzero_by_channel'][28])
        rows.append([SPLIT[split],*[f'{v*100:.2f}%' for v in values]])
    sections.append(table(['分布','曾移动','曾有长距离移动','曾反向移动','曾重访位置','己方棋子局部密度非零'],rows))
    rows = []
    for prefix, label in [('training', '1,024 对局'), ('scale', '5,120 对局')]:
        for split in SPLITS:
            for group in ('opening_zero', 'moved', 'never_moved', 'late', 'moved_after_128'):
                scores = result['group_means'][prefix]['baseline'][split]
                if group not in scores:
                    continue
                base = scores[group]
                values = [result['paired_group_comparisons'][prefix][arm][split][group]['nll']
                          for arm in ('temporal', 'relational')]
                rows.append([label, SPLIT[split], GROUP[group], base['games'], base['labels'],
                             interval(values[0]), interval(values[1])])
    sections.append(table(['训练数据', '分布', '分组', '测试对局数', '有效标签数/种子', '轨迹 NLL 差及区间', '关系 NLL 差及区间'], rows))
    sections.append('后期组即使区间很窄，也必须结合实际对局数阅读。来自一两个对局的大量标签，不能证明对整个长局面分布的效果。额外随机行为测试提供不同动作分布，但不代表所有人类或训练中策略。')

    sections += ['## 简单先验、信息边界与特征依赖',
        '空间关系特征完全由已有公开观测计算得到，因此不增加隐藏信息。长期轨迹保留更长的公开历史，可能提供现有短历史之外的统计线索。两者是否有用，最终由新对局上的概率预测成绩判断。']
    rows = []
    for split in SPLITS:
        refs = primary['references'][split]
        tabular = read(root / 'tabular_diagnostics.json')['evaluations'][split]
        rows.append([SPLIT[split], f"{refs['rule']['nll']:.5f}", f"{refs['frequency']['nll']:.5f}",
                     *[f"{tabular[name]['all']['nll']:.5f}" for name in ('slot_initial', 'slot_all', 'slot_move')],
                     f"{primary['primary_means']['baseline'][split]['nll']:.5f}",
                     f"{expanded['primary_means']['baseline'][split]['nll']:.5f}"])
    sections += [table(['分布', '规则候选先验', '训练类型频率', '初始位置频率', '全时段位置频率',
                        '位置＋移动分桶', '1,024 对局基线', '5,120 对局基线'], rows),
        '三个位置表仅由原 1,024 个训练对局拟合，采用预先固定的加一平滑和公开候选遮罩，没有在测试集拟合。它们属于事后解释性参照；其中与 5,120 对局网络的比较不是等数据量模型竞赛。']
    prior = result['opening_distribution_reference']
    sections.append(f"按当前随机布阵生成器推导的初始位置边际先验，其平均熵为 {prior['mean_opening_marginal_entropy_nats']:.6f} nats，"
                    f"最高概率类型的平均概率为 {prior['mean_opening_marginal_top1'] * 100:.4f}%。"
                    '这是该理想随机布阵分布的初始局面参考，不能当作整局游戏、当前有限数据集或所有对手策略的不可突破误差下界。')
    sections.append('独立构造已验证：交换敌方两个合法位置上的不同隐藏类型，可以得到完整 317 通道观测、公开 global 信息、两类新增特征都完全相同而监督身份不同的初始局面。因此，普遍完美识别不是合理目标；合理目标是利用可见信息输出更准确、校准更好的条件概率。这个反例本身不量化当前网络距离最优条件概率还有多远。')
    rows = []
    for label, report in [('1,024 对局', primary), ('5,120 对局', expanded)]:
        for arm in ('temporal', 'relational'):
            for key, control in [('zero', '新增特征置零'), ('shuffle', '跨局面打乱新增特征')]:
                value = report['feature_control_minus_full'][arm][key]
                rows.append([label, ARM[arm], control, f"{value['nll']:+.6f}", f"{value['accuracy'] * 100:+.4f}"])
    sections += [table(['数据', '模型', '推理时干预', '相对正常输入的 NLL 变化', '准确率变化（百分点）'], rows),
        '推理时干预会使输入偏离训练分布，可检验模型对新增特征的依赖；它不能单独证明特征带来泛化收益，也不能代替配对训练基线。']

    sections += ['## 早期选模分辨率的探索性检查']
    if early:
        report=early['early_analysis']
        sections += ['两个方向各 6/6 个单元完成，每单元 1,000 步，每 50 步验证。使用原 1,024 个对局、相同初始化与优化设置，在第 500、1,000 步核对全部验证指标完全一致，合并的小批量损失也与主实验对应区间一致；选择同一步时，实际权重亦一致。',
                     '此检查判断较粗的每 500 步验证是否漏掉早期最优模型。它不是新的数据量实验，也不是新增的独立种子证据。']
        rows=[]
        for arm in ARM:
            chosen=early['selected_steps'][arm]['dense_steps']
            for split in SPLITS:
                score=report['primary_means'][arm][split]
                delta=early['dense_selection_minus_primary_selection'][arm][split]['nll']
                feature='—' if arm=='baseline' else interval(report['paired_comparisons'][arm][split]['nll'])
                rows.append([ARM[arm],SPLIT[split],', '.join(map(str,chosen)),f"{score['nll']:.6f}",
                             f"{score['accuracy']*100:.3f}%",interval(delta),feature])
        sections += [table(['输入','分布','选中步骤','NLL','准确率','相对原选模的 NLL 变化','相对密集选模基线的 NLL 差'],rows),
                     '![早期学习曲线](../analysis_early/belief_feature_results.png)',
                     '完整数值与重跑一致性核对见 [早期复核](../analysis_early/resolution_check.json)。']
    else:
        counts={role:len(list((root/f'early_{role}').glob('*/done'))) for role in 'AB'}
        sections.append(f"该补充项未形成完整分析：A 已完成 {counts['A']}/6，B 已完成 {counts['B']}/6。没有据此给出完整配对结论；原主实验、扩充复核和 26 个模型的重评分结果独立有效。退出状态与部分文件保留在 early_A/B。")

    extra = primary['supplementary_seed_604']
    if extra:
        rows = []
        for split in SPLITS:
            a, b = [extra[arm]['evaluations'][split] for arm in ('baseline', 'relational')]
            rows.append([SPLIT[split], f"{a['nll']:.6f}", f"{b['nll']:.6f}", f"{b['nll'] - a['nll']:+.6f}"])
        sections += ['## 关系特征的补充种子 604', table(['分布', '基线 NLL', '关系 NLL', '差值'], rows),
                     '此种子不混入两个特征方向的三种子主比较；单个补充种子用于检查方向一致性。']

    maximum = max(x['maximum_absolute_metric_difference'] for x in result['independent_rescoring_agreement'])
    sections += ['## 校验、来源和未验证范围',
        f"主实验与扩充复核的 26 个已保存模型重新评分，五项总体指标与原始评估记录的最大绝对差为 {maximum:.3g}，低于检查阈值 2e-6。"
        '这是另一路指标汇总代码对同一模型和数据的复算；它不等于独立团队复现。',
        '训练前校验数据哈希及完整对局 ID 分区。原始 8 个、扩充后 24 个分片全部在本地通过哈希校验；扩充训练包含 5,120 个独立对局，验证与三种测试各为 128 个独立对局。所有已完成训练单元保存最优及最后原始参数、优化器、配置、曲线、结果和完成标记。主实验、扩充复核和保存模型诊断的整体退出码均为 0；早期补充项的完成状态单独列示。',
        '本地完整回归最近一次为 1,392 项通过、171 项跳过、70 条警告；后续涵盖分析、公开移动分组及完整文件清单校验的定向检查为 15 项通过。两台服务器各通过 16 项原生 GPU 检查，并在正式训练前完成相同网络的 80 步执行预检。跳过项及既有警告没有被计作通过。',
        table(['用途', '固定提交'], [
            ['主训练', '57940a09dd3ad0b24534a827e0c7b3ffda252ffd'],
            ['扩充数据与复核训练', 'd958f70b9822a711354f1eb047fdb02f09a26f04'],
            ['保存模型诊断', '0c10407d1c58e3cd5888facb46b49e093c5c1a93'],
            ['汇总分析', result['analysis_commit']]]),
        'A 的训练硬截止为北京时间 08:20，B 为 09:20，分别比释放时间提前 40 分钟。启动时一次性计算截止计时器；阶段衔接和本地回收只响应完成标记。本轮代码和结果保留在本地，没有推送 GitHub。',
        '尚未验证：将新特征接回在线 belief／策略联合训练后的稳定性、搜索价值或实际胜率；所有长局面和对手分布；全局联合隐藏布局的建模收益。本轮特征对照只支持固定策略生成数据上的离线条件身份预测结论，前文原始行棋策略复测不能归因于新特征。',
        '逐种子与逐对局统计见 [完整汇总](synthesis.json)、[主实验明细](../analysis_training/analysis.json)、'
        '[扩充实验明细](../analysis_scale/analysis.json)；协议见同目录 PROTOCOL.md。']
    return '\n\n'.join(sections) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = read(args.root / 'analysis_final/synthesis.json')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(make_report(args.root, result))
    print(json.dumps({'report': str(args.output), 'source_status': result['status']}))
