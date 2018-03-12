import { Rule, RuleGroup, isRuleGroup, Condition } from './ruleModel';

import * as nt from '../service/num';

export function groupRules(rules: Rule[]): Rule | RuleGroup {
  if (rules.length === 0) throw 'The length of the rules to be grouped should be at least 1';
  if (rules.length === 1) {
    let _ret = {...(rules[0]), rules};
    _ret.rules[0].parent = _ret;
    return _ret;
  }
  let nested: Rule[] = [];
  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i];
    if (isRuleGroup(rule)) {
      nested = nested.concat(rule.rules);
    } else {
      nested.push(rule);
    }
  }
  const supports = nested.map(rule => rule.support);
  let support: number[] | number[][];
  let supportSums: number[];
  let _support: number[];
  if (Array.isArray(supports[0][0])) {
    support = nt.sumMat(supports as number[][][]);
    supportSums = (<number[][][]> supports).map(s => nt.sum(nt.sumVec(s)));
    _support = nt.sumVec(support);
  } else {
    support = nt.sumVec(supports as number[][]);
    supportSums = (<number[][]> supports).map(s => nt.sum(s));
    _support = support;
  }
  const totalSupport = nt.sum(supportSums);
  const output = nt.sumVec(nested.map(r => nt.muls(r.output, r.totalSupport / totalSupport)));
  const label = nt.argMax(output);
  const conditions: Condition[] = [];
  rules.forEach((r, i) => {
    const conds = r.conditions.map((c) => ({...c, rank: i}));
    conditions.push(...conds);
  });
  const ret = { rules, support, _support, output, label, totalSupport, conditions, idx: rules[0].idx };
  rules.forEach((r) => r.parent = ret);
  return ret;
}

export function groupBySupport(rules: Rule[], minSupport: number = 0.01): Rule[] {
  const retRules: Rule[] = new Array();

  // let prevSum = 0.;
  let tmpRules: Rule[] = [];
  for (let i = 0; i < rules.length; i++) {
    const rule = rules[i];
    if (rule.totalSupport >= minSupport) {
      if (tmpRules.length > 0) {
        retRules.push(groupRules(tmpRules));
        tmpRules = [];
        // prevSum = 0.;
      }
      retRules.push(rule);
    } else {
      tmpRules.push(rule);
      // prevSum += rule.totalSupport;
    }
  }
  if (tmpRules.length) {
    retRules.push(groupRules(tmpRules));
  }
  return retRules;
}