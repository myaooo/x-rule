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
  name: DataTypeX;
  ranges: [number, number][];
  categories: (string[] | null)[];
  ratios: number[][];
  readonly discretizers: Discretizer[];
}

export type StreamLayer = Int32Array;

export type Stream = StreamLayer[];

export type Streams = Stream[];

export type ConditionalStreams = Streams[];

export function createStreams(raw: number[][][]): Streams {
  return raw.map((rawStream: number[][]) => rawStream.map((layer: number[]) => new Int32Array(layer)));
}

export function createConditionalStreams(raw: number[][][][]): ConditionalStreams {
  return raw.map((streams: number[][][]) => createStreams(streams));
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
  public name: DataTypeX;
  public ratios: number[][];
  public ranges: [number, number][];
  public categories: (string[] | null)[];
  public isCategorical: boolean[];
  public discretizers: Discretizer[];
  public streams?: Streams;
  public conditionalStreams?: ConditionalStreams;
  constructor(raw: PlainData) {
    const { data, target, featureNames, labelNames, hists, name, ranges } = raw;
    const { categories, discretizers, ratios, isCategorical } = raw;
    this.data = data.map((d: number[]) => new Float32Array(d));
    this.target = new Int32Array(target);
    this.featureNames = featureNames;
    this.labelNames = labelNames;
    this.hists = hists;
    this.name = name;
    this.ranges = ranges;
    this.categories = categories;
    this.ratios = ratios;
    this.discretizers = discretizers;
    this.isCategorical = isCategorical;
    // this.categorical = categorical;
  }
  public categoryInterval(f: number, c: number): [number | null, number | null] {
    const intervals = this.discretizers[f].intervals;
    return intervals ? intervals[c] : [null, null];
  }
  public categoryDescription(f: number, c: number, maxLength: number = 20, abr: boolean = false): string {
    const {featureNames, discretizers, categories} = this;
    const cutSize = Math.round((maxLength - 2) / 2);
    const featureName = featureNames[f];
    const intervals = discretizers[f].intervals;
    const category = intervals ? intervals[c] : c;
    let featureMap = (feature: string): string => `${feature} is any`;
    if (typeof category === 'number' && categories) {
      featureMap = (feature: string) => `${feature} = ${(<string[]> categories[f])[c]}`;
    } else {
      const low = category[0];
      const high = category[1];
      if (low === null && high === null) featureMap = (feature: string) => `${feature} is any`;
      else {
        const lowString = low !== null ? `${low.toPrecision(3)} < ` : '';
        const highString = high !== null ? ` < ${high.toPrecision(3)}` : '';
        featureMap = (feature: string) => lowString + feature + highString;
      }
    }
    if (abr) {
      const abrString = featureName.length > maxLength
      ? `"${featureName.substr(0, cutSize)}…${featureName.substr(-cutSize, cutSize)}"`
      : featureName;
      return featureMap(abrString);
    }
    return featureMap(featureName);
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

export type Support = number[][];

export type SupportMat = number[][][];

export type SupportType = Support | SupportMat;

export function isSupportMat(support: SupportType): support is SupportMat {
  return Array.isArray(support[0][0]);
}