import FeatureList from '../components/FeatureList';
import { Action } from 'redux';
import { connect } from 'react-redux';
import { RootState, getActivatedFeature, Dispatch, selectFeature, getFeatureIsSelected } from '../store';

type FeatureListStateProp = {
  activatedFeature?: number;
  featureIsSelected?: boolean;
};

const mapStateToProps = (state: RootState): FeatureListStateProp => {
  return {
    activatedFeature: getActivatedFeature(state),
    featureIsSelected: getFeatureIsSelected(state)
  };
};

type FeatureListDispatchProp = {
  selectFeature?: ({ idx, deselect }: { idx: number; deselect: boolean }) => Action;
};

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): FeatureListDispatchProp => {
  return {
    selectFeature: ({ idx, deselect }: { idx: number; deselect: boolean }): Action =>
      dispatch(selectFeature({idx, deselect}))
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(FeatureList);
