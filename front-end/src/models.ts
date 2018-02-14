export interface ModelBase {
    readonly type: string;
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

export interface LeafNode {
    readonly value: number[];
    readonly impurity: number;
    collapsed?: boolean;
    readonly idx: number;
    readonly output: number;
}

export interface InternalNode extends LeafNode {
    readonly left: TreeNode;
    readonly right: TreeNode;
    readonly feature: number;
    readonly threshold: number;
}

export type TreeNode = InternalNode | LeafNode;

export interface TreeModel extends ModelBase {
    readonly type: 'tree';
    readonly root: TreeNode;
    readonly nNodes: number;
    readonly maxDepth: number;
}

export function isLeafNode(node: TreeNode): node is LeafNode {
    return (<InternalNode> node).left === undefined;
}

export function isInternalNode(node: TreeNode): node is InternalNode {
    return (<InternalNode> node).left !== undefined;
}

export function isTreeModel(model: ModelBase): model is TreeModel {
    return model.type === 'tree';
}

export function isSurrogate(model: ModelBase): model is Surrogate {
    return (<Surrogate> model).target !== undefined;
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
