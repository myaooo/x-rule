import { ModelBase, DataSet, DataTypeX } from '../models';

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
}

export const initTreeStyles: TreeStyles = {
  linkWidth: 1.0,
};

export interface RuleStyles {
  size: number;
  width: number;
  mode: string;
}

export const initRuleStyles: RuleStyles = {
  size: 30,
  mode: 'list',
  width: 50,
};

export interface RootState {
  // modelBase: ModelBaseState;
  // selectedModel: string;
  model: ModelState;
  dataBase: DataBaseState;
  selectedData: DataTypeX[];
  selectedFeatures: FeatureState[];
  treeStyles: TreeStyles;
  ruleStyles: RuleStyles;
}

// const initialState: State = {
//     model: null,
// };
