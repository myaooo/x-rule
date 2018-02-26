import * as React from "react";

type TransitionValue = number | number[] | string[] | [string, string] | CustomInterpolator;

/**
 * Timing must be placed on the "timing" key in the transition.
 */
export interface Timing {
  delay?: number;
  duration?: number;
  ease?: (t: number) => number;
}

/**
 * Events must be placed on the "events" key in the transition.
 */
export interface Events {
  start?: () => void;
  interrupt?: () => void;
  end?: () => void;
}

export interface CustomInterpolator {
  (t: number): any;
}

export interface NameSpace {
  [key: string]: TransitionValue;
}

export interface Transition {
  [key: string]: TransitionValue | NameSpace | Events | Timing;
}

export interface TransitionFunction {
  (): Transition | Array<Transition>;
}

export interface PlainObject {
  [key: string]: number | string | PlainObject;
}

export interface PlainObjectFunction {
  (): PlainObject;
}
