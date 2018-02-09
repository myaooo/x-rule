// import { createSelector } from 'reselect';
import { ModelBase, PlainData } from '../models';
import { RootState, SelectedDataType } from './state';
import { DataBaseState } from './index';

export const getModel = (state: RootState): ModelBase | null => state.model.model;
export const getModelIsFetching = (state: RootState): boolean => state.model.isFetching;
export const getSelectedDataNames = (state: RootState): SelectedDataType[] => state.selectedData;
export const getSelectedData = (state: RootState): (PlainData | undefined)[] => {
  return state.selectedData.map((dataName: SelectedDataType) => state.dataBase[dataName]);
};
export const getData = (state: RootState): DataBaseState => state.dataBase;
export const getTrainData = (state: RootState): PlainData | undefined => state.dataBase.train;
export const getTestData = (state: RootState): PlainData | undefined => state.dataBase.test;
export const getActivatedFeature = (state: RootState): number => 
  (state.selectedFeature.count > 0 ? state.selectedFeature.idx : -1);
export const getFeatureIsSelected = (state: RootState): boolean => 
  (state.selectedFeature.count === 2);