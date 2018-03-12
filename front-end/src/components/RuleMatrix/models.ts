import { Rule, Condition } from '../../models';
import { Stream } from '../../models/data';

export interface ConditionX extends Condition {
  title: string;
  interval: [number | null, number | null];
  // activeRatio: [number, number];
  expanded?: boolean;
  stream?: Stream;
  x: number;
  width: number;
}

export interface RuleX extends Rule {
  conditions: ConditionX[];
  x: number;
  y: number;
  height: number;
  width: number;
  // support: number[];
  // totalSupport: number;
  // collapsed?: boolean;
}

// export class RuleX implements RuleX {
//   constructor
// }