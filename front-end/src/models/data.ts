// DataSets

export type DataType = 'train' | 'test';
export type SurrogateDataType = 'sample train' | 'sample test';
export type DataTypeX = DataType | SurrogateDataType;

export type PlainMatrix = number[][];

export interface Discretizer {
  readonly cutPoints: number[] | null;
  readonly intervals: [number | null, number | null][] | null;
  readonly min: number;
  readonly max: number;
  readonly ratios: number[];
}

export interface Histogram {
    counts: number[];
    centers: number[];
}

export interface PlainData {
    data: number[][];
    target: number[];
    featureNames: string[];
    labelNames: string[];
    isCategorical: boolean[];
    hists: Histogram[];
    name: 'train' | 'test';
    ranges: [number, number][];
    categories?: number[];
    readonly discretizers: Discretizer[];
}

export type StreamLayer = Int32Array;

export type Stream = StreamLayer[];

export type Streams = Stream[];

export type ConditionalStreams = Streams[];

export function createStreams(raw: number[][][]): Streams {
  return raw.map((rawStream: number[][]) => (
    (rawStream.map((layer: number[]) => new Int32Array(layer)))
  ));
}

export function createConditionalStreams(raw: number[][][][]): ConditionalStreams {
  return raw.map((streams: number[][][]) => (
    createStreams(streams)
  ));
}

export function isConditionalStreams(streams: Streams | ConditionalStreams): streams is ConditionalStreams {
    return streams[0][0][0] instanceof Int32Array;
}

export class DataSet {
    public data: Float32Array[];
    public target: Int32Array;
    public featureNames: string[];
    public labelNames: string[];
    public hists: Histogram[];
    public name: DataType;
    public ranges: [number, number][];
    public categories: number[];
    public discretizers: Discretizer[];
    public streams?: Streams;
    public conditionalStreams?: ConditionalStreams;
    constructor(raw: PlainData) {
        const {data, target, featureNames, labelNames, hists, name, ranges, categories, discretizers} = raw;
        this.data = data.map((d: number[]) => new Float32Array(d));
        this.target = new Int32Array(target);
        this.featureNames = featureNames;
        this.labelNames = labelNames;
        this.hists = hists;
        this.name = name;
        this.ranges = ranges;
        this.categories = categories ? categories : featureNames.map((f) => 0);
        this.discretizers = discretizers;
        // this.categorical = categorical;
    }
    public categoryInterval(f: number, c: number): [number | null, number | null] {
        const intervals = this.discretizers[f].intervals;
        return intervals ? intervals[c] : [null, null];
    }

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