import { createSelector } from 'reselect';
import { ModelBase, DataSet, DataTypeX } from '../models';
import { RootState, TreeStyles, FeatureState, FeatureStatus } from './state';
import { DataBaseState } from './index';

export const getModel = (state: RootState): ModelBase | null => state.model.model;
export const getModelIsFetching = (state: RootState): boolean => state.model.isFetching;
export const getSelectedDataNames = (state: RootState): DataTypeX[] => state.selectedData;
export const getSelectedData = (state: RootState): (DataSet | undefined)[] => {
  const ret: (DataSet | undefined)[] = [];
  for (let data of state.selectedData) {
    if (data in state.dataBase) ret.push(state.dataBase[data]);
  }
  return ret;
  // return state.selectedData.map((dataName: DataTypeX) => state.dataBase[dataName]);
};
export const getData = (state: RootState): DataBaseState => state.dataBase;
export const getTrainData = (state: RootState): DataSet | undefined => state.dataBase.train;
export const getTestData = (state: RootState): DataSet | undefined => state.dataBase.test;

// export const getActivatedFeatures = (state: RootState): number[] => {
//   return state.selectedFeatures
//     .filter((f: FeatureState) => f.status === FeatureStatus.HOVER)
//     .map((f: FeatureState) => f.idx);
// };
// export const getSelectedFeatures = (state: RootState): boolean => 
//   (state.selectedFeature.count === 2);

export const getTreeStyles = (state: RootState): TreeStyles => (state.treeStyles);

export const getFeatureStates = (state: RootState): FeatureState[] =>  
  (state.selectedFeatures);

export const getActivatedFeatures = createSelector(
  [getFeatureStates],
  (featureStates: FeatureState[]): number[] => {
    return featureStates.filter((f: FeatureState) => f.status === FeatureStatus.HOVER)
    .map((f: FeatureState) => f.idx);
  }
);

export const getSelectedFeatures = createSelector(
  [getFeatureStates],
  (featureStates: FeatureState[]): number[] => {
    return featureStates.filter((f: FeatureState) => f.status === FeatureStatus.SELECT)
    .map((f: FeatureState) => f.idx);
  }
);