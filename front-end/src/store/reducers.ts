import { combineReducers } from 'redux';
import { isRuleModel, isTreeModel, RuleList, DataSet, DataTypeX } from '../models';
import {
  ModelState,
  DataBaseState,
  FeatureState,
  TreeStyles,
  initTreeStyles,
  FeatureStatus,
  RuleStyles,
  initRuleStyles,
  RootState
} from './state';
import { collapseInit } from '../service/utils';
import { ReceiveStreamAction } from './actions';
import { ConditionalStreams, Streams } from '../models/stream';

import {
  ReceiveSupportAction,
  ActionType,
  RequestModelAction,
  ReceiveModelAction,
  // RequestDatasetAction,
  ReceiveDatasetAction,
  SelectDatasetAction,
  SelectFeatureAction,
  ChangeTreeStylesAction,
  ChangeRuleStylesAction
} from './actions';

export const initialModelState: ModelState = {
  model: null,
  isFetching: false
};

// export const initialModelBaseState: ModelBaseState = {};

export const initialDataBaseState: DataBaseState = {};

export const initialFeaturesState: FeatureState[] = [];

function modelStateReducer(
  state: ModelState = initialModelState,
  action: RequestModelAction | ReceiveModelAction | ReceiveSupportAction
): ModelState {
  switch (action.type) {
    case ActionType.REQUEST_MODEL:
      // console.log("start Fetching...");  // tslint:disable-line
      return { ...state, isFetching: true };
    case ActionType.RECEIVE_MODEL:
      // console.log("receiving model...");  // tslint:disable-line
      let model = action.model;
      if (model !== null) {
        if (isRuleModel(model)) model = new RuleList(model);
        if (isTreeModel(model)) collapseInit(model.root);
      }
      return {
        isFetching: false,
        model
      };
    case ActionType.RECEIVE_SUPPORT:
      const aModel = state.model;
      if (aModel instanceof RuleList) {
        aModel.support(action.support);
      }
      return {
        isFetching: false,
        model: aModel
      };
    default:
      return state;
  }
}

function dataBaseReducer(
  state: DataBaseState = initialDataBaseState,
  action: ReceiveDatasetAction | ReceiveStreamAction
): DataBaseState {
  switch (action.type) {
    case ActionType.RECEIVE_DATASET:
      const newState: DataBaseState = {};
      newState[action.dataType] = new DataSet(action.data);
      return { ...state, ...newState };

    case ActionType.RECEIVE_STREAM:
      const newState2: DataBaseState = {};
      const dataset = state[action.dataType];
      if (dataset) {
        if (action.conditional) {
          dataset.conditionalStreams = action.streams as ConditionalStreams;
        } else {
          dataset.streams = action.streams as Streams;
        }
        newState2[action.dataType] = dataset;
        return { ...state, ...newState2 };
      }
      return state;

    default:
      return state;
  }
}

function selectDatasetReducer(state: DataTypeX[] = [], action: SelectDatasetAction): DataTypeX[] {
  switch (action.type) {
    case ActionType.SELECT_DATASET:
      return action.dataNames;
    default:
      return state;
  }
}

function selectedFeaturesReducer(
  state: FeatureState[] = initialFeaturesState,
  action: SelectFeatureAction
): FeatureState[] {
  switch (action.type) {
    case ActionType.SELECT_FEATURE:
      if (action.deselect) {
        const idx = state.findIndex((f: FeatureState) => f.idx === action.idx);
        if (idx === -1) return state;
        const feature = state[idx];
        feature.status--;
        if (feature.status < FeatureStatus.HOVER) {
          return [...state.slice(0, idx), ...state.slice(idx + 1)];
        }
        return state;
      } else {
        const idx = state.findIndex((f: FeatureState) => f.idx === action.idx);
        if (idx === -1) return [...state, { idx: action.idx, status: FeatureStatus.HOVER }];
        else if (state[idx].status < FeatureStatus.SELECT) {
          state[idx].status++;
        }
        return state;
      }
    default:
      return state;
  }
}

function treeStyleReducer(state: TreeStyles = initTreeStyles, action: ChangeTreeStylesAction): TreeStyles {
  switch (action.type) {
    case ActionType.CHANGE_TREE_STYLES:
      return { ...state, ...action.newStyles };
    default:
      return state;
  }
}

function ruleStyleReducer(state: RuleStyles = initRuleStyles, action: ChangeRuleStylesAction): RuleStyles {
  switch (action.type) {
    case ActionType.CHANGE_RULE_STYLES:
      return { ...state, ...action.newStyles };
    default:
      return state;
  }
}
// function selectedDataReducer(
//   state: string,
//   action:
// )

export const rootReducer = combineReducers<RootState>({
  model: modelStateReducer,
  dataBase: dataBaseReducer,
  selectedData: selectDatasetReducer,
  selectedFeatures: selectedFeaturesReducer,
  treeStyles: treeStyleReducer,
  ruleStyles: ruleStyleReducer
});
