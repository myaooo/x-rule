import { ModelBase, PlainData } from '../models';

export interface ModelState {
  readonly model: ModelBase | null;
  readonly isFetching: boolean;
}

// export interface DataState {
//   readonly data: PlainData | null;
//   readonly isFetching: boolean;
// }

export interface DataBaseState {
  train?: PlainData;
  test?: PlainData;
}

export type ModelBaseState = { [modelName: string]: ModelBase };

export interface FeatureState {
  idx: number;  // the idx of the feature
  count: number;  // 0: no feature selected; 1: mouse enter 2: selected
}

// export const initialDataState: DataState = {
//   data: null,
//   isFetching: false
// };

export type SelectedDataType = 'train' | 'test' | null;

export interface RootState {
  // modelBase: ModelBaseState;
  // selectedModel: string;
  model: ModelState;
  dataBase: DataBaseState;
  selectedData: SelectedDataType;
  selectedFeature: FeatureState;
}

// const initialState: State = {
//     model: null,
// };
