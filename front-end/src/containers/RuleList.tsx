import RuleList from '../components/RuleList';
import { Action } from 'redux';
import { connect } from 'react-redux';
import { 
  RootState, Dispatch, selectFeature, FeatureStatus, getFeatureStates, FeatureState 
} from '../store';

type RuleListStateProp = {
  featureStatus(i: number): FeatureStatus,
};

const mapStateToProps = (state: RootState): RuleListStateProp => {
  return {
    featureStatus: (i: number) => {
      const f = getFeatureStates(state).find((v: FeatureState) => v.idx === i);
      return f ? f.status : FeatureStatus.DEFAULT;
    }
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
