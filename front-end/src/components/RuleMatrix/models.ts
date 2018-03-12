import { Rule, Condition } from '../../models';
import { Stream } from '../../models/data';

export interface ConditionX extends Condition {
  title: string;
  desc: string;
  interval: [number, number];
  histRange: [number, number];
  // activeRatio: [number, number];
  expanded?: boolean;
  stream?: Stream;

  x: number;
  width: number;
  height: number;
}

export interface RuleX extends Rule {
  conditions: ConditionX[];
  x: number;
  y: number;
  height: number;
  width: number;
  expanded: boolean;
  // support?: number[] | number[][];
  // support: number[];
  // totalSupport: number;
  // collapsed?: boolean;
}

export interface Feature {
  text: string;
  x: number;
  width: number;
  count: number;
  cutPoints?: number[] | null;
  range?: [number, number];
  expanded?: boolean;
}

// export class RuleX implements RuleX {
//   constructor
// }