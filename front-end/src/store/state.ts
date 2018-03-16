import { ModelBase, DataSet, DataTypeX, Streams, ConditionalStreams } from '../models';

export interface ModelState {
  readonly model: ModelBase | null;
  readonly isFetching: boolean;
}

// export interface DataState {
//   readonly data: PlainData | null;
//   readonly isFetching: boolean;
// }

export type DataBaseState = {
  [name in DataTypeX]?: DataSet;
};

export type StreamBaseState = {
  [name in DataTypeX]?: {streams?: Streams, conditionalStreams?: ConditionalStreams};
};

// export type ModelBaseState = { [modelName: string]: ModelBase };

export enum FeatureStatus {
  DEFAULT = 0,
  HOVER = 1,
  SELECT = 2,
}

export interface FeatureState {
  idx: number;  // the idx of the feature
  status: FeatureStatus;  // 0: no feature selected; 1: mouse enter 2: selected
}

// export const initialDataState: DataState = {
//   data: null,
//   isFetching: false
// };

export interface TreeStyles {
  linkWidth: number;
  // conditional: boolean;
}

export const initTreeStyles: TreeStyles = {
  linkWidth: 1.0,
  // conditional: false,
};

export interface RuleStyles {
  flowWidth: number;
  rectWidth: number;
  rectHeight: number;
  mode: 'list' | 'matrix';
  // conditional: boolean;
}

export interface SupportState {
  isFetching: boolean;
  support: number[][] | number[][][] | null;
}

export const initSupportState: SupportState = {
  isFetching: false,
  support: null,
};

export const initRuleStyles: RuleStyles = {
  flowWidth: 50,
  mode: 'matrix',
  rectWidth: 40,
  rectHeight: 30,
  // conditional: false,
};

export interface Settings {
  conditional: boolean;
  supportMat: boolean;
  minSupport: number;
}

export const initialSettings: Settings = {
  conditional: false,
  supportMat: true,
  minSupport: 0.01,
};

export type DataFilter = number[] | null;

export const initDataFilter: DataFilter[] = [];

export const initialStreamBaseState: StreamBaseState = {};

export interface RootState {
  // modelBase: ModelBaseState;
  // selectedModel: string;
  model: ModelState;
  dataBase: DataBaseState;
  dataFilters: DataFilter[];
  streamBase: StreamBaseState;
  selectedData: DataTypeX[];
  selectedFeatures: FeatureState[];
  treeStyles: TreeStyles;
  ruleStyles: RuleStyles;
  settings: Settings;
  support: SupportState;
  // conditional: boolean;
}

// const initialState: State = {
//     model: null,
// };
