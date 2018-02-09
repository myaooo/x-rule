
export interface Condition {
    readonly feature: number;
    readonly category: number;
    // readonly support: number;
}

export interface Rule {
    readonly conditions: Condition[];
    readonly output: number[];
    support: number[];
}

export interface ModelBase {
    readonly type: string;
    readonly dataset: string;
    [propName: string]: any;
    // predict(x: Float32Array | Int32Array): Promise<number>;
    // predictProb(x: Float32Array | Int32Array): Promise<Float32Array>;
}

export interface Discretizer {
    readonly cutPoints: number[] | null;
    readonly intervals: (number | null)[][] | null;
    readonly min: number;
    readonly max: number;
    readonly ratios: number[];
}

export interface RuleModel extends ModelBase {
    readonly type: 'rule';
    readonly rules: Rule[];
    readonly discretizers: Discretizer[];
}

export function isRuleModel(model: ModelBase): model is RuleModel {
    return model.type === 'rule';
}

export type PlainMatrix = number[][];

export interface Histogram {
    counts: number[];
    centers: number[];
}

export interface PlainData {
    data: number[][];
    target: number[];
    featureNames: string[];
    labelNames: string[];
    continuous: boolean[];
    hists: Histogram[];
    name: 'train' | 'test';
}

export class Matrix {
    data: Float32Array;
    size1: number;
    size2: number;
    constructor(size1: number, size2: number) {
        this.data = new Float32Array(size1 * size2);
    }
    // get_column()
}

export interface NeuralNetwork extends ModelBase {
    readonly neurons: Int32Array;
    readonly weights: Matrix[];
    readonly bias: Float32Array[];
}
