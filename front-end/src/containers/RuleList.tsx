import RuleList from '../components/RuleList';
import { Action } from 'redux';
import { connect } from 'react-redux';
import { RootState, getActivatedFeature, Dispatch, selectFeature, getFeatureIsSelected } from '../store';

type RuleListStateProp = {
  activatedFeature: number;
  featureIsSelected: boolean;
};

const mapStateToProps = (state: RootState): RuleListStateProp => {
  return {
    activatedFeature: getActivatedFeature(state),
    featureIsSelected: getFeatureIsSelected(state)
  };
};

type RuleListDispatchProp = {
  selectFeature: ({ idx, deselect }: { idx: number; deselect: boolean }) => Action;
};

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): RuleListDispatchProp => {
  return {
    selectFeature: ({ idx, deselect }: { idx: number; deselect: boolean }): Action =>
      dispatch(selectFeature({idx, deselect}))
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(RuleList);
