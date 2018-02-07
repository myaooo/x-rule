// import { createSelector } from 'reselect';
import { ModelBase, PlainData } from '../models';
import { RootState, SelectedDataType } from './state';

export const getModel = (state: RootState): ModelBase | null => state.model.model;
export const getModelIsFetching = (state: RootState): boolean => state.model.isFetching;
export const getSelectedDataName = (state: RootState): SelectedDataType => state.selectedData;
export const getData = (state: RootState): PlainData | undefined => {
  if (state.selectedData === null)
    return undefined;
  return state.dataBase[state.selectedData];
};
export const getTrainData = (state: RootState): PlainData | undefined => state.dataBase.train;
export const getTestData = (state: RootState): PlainData | undefined => state.dataBase.test;
export const getActivatedFeature = (state: RootState): number => 
  (state.selectedFeature.count > 0 ? state.selectedFeature.idx : -1);
export const getFeatureIsSelected = (state: RootState): boolean => 
  (state.selectedFeature.count === 2);