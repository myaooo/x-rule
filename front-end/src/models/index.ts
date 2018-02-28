export interface ModelBase {
  readonly type: string;
  readonly name: string;
  readonly dataset: string;
  readonly nFeatures: number;
  readonly nClasses: number;
  [propName: string]: any;
  // predict(x: Float32Array | Int32Array): Promise<number>;
  // predictProb(x: Float32Array | Int32Array): Promise<Float32Array>;
}

export interface Surrogate extends ModelBase {
  readonly target: string;  // the name of the target model
}

export function isSurrogate(model: ModelBase): model is Surrogate {
  return (<Surrogate> model).target !== undefined;
}

export * from './data';
export * from './ruleModel';
export * from './tree';
