import * as React from 'react';
// import * as d3 from 'd3';
import { Card, Divider } from 'antd';
import { connect } from 'react-redux';
import {
  Dispatch,
  fetchModelIfNeeded,
  getModel,
  RootState,
  getSelectedData,
  getModelIsFetching
} from '../store';
import { RuleModel, PlainData, ModelBase, isRuleModel } from '../models';
// import Tree from '../components/Tree';
import RuleList from '../containers/RuleList';
import FeatureList from '../containers/FeatureList';

export interface ModelViewStateProp {
  model: RuleModel | ModelBase | null;
  modelIsFetching: boolean;
  data: (PlainData | undefined)[];
}

const mapStateToProps = (state: RootState): ModelViewStateProp => {
  return {
    model: getModel(state),
    modelIsFetching: getModelIsFetching(state),
    data: getSelectedData(state)
  };
};

export interface ModelViewDispatchProp {
  loadModel: (modelName: string) => Dispatch;
}

const mapDispatchToProps = (dispatch: Dispatch, ownProps: any): ModelViewDispatchProp => {
  return {
    // loadModel: bindActionCreators(getModel, dispatch),
    loadModel: (modelName: string): Dispatch => dispatch(fetchModelIfNeeded(modelName)),
  };
};

export interface ModelViewProp
  extends 
  ModelViewStateProp,
  ModelViewDispatchProp { 
    modelName: string;
  }

class ModelView extends React.Component<ModelViewProp, any> {
  svgRef: SVGSVGElement;
  constructor(props: ModelViewProp) {
    super(props);
  }
  componentDidMount() {
    const { loadModel, modelName } = this.props;
    loadModel(modelName);
  }
  // componentDidUpdate() {
  //   const { width, height } = {width: 960, height: 720};
  //   d3.select(this.svgRef).attr('width', width).attr('height', height);
  // }
  render(): React.ReactNode {
    const { modelIsFetching, model, data, modelName } = this.props;
    if (model === null || !isRuleModel(model)) {
      if (modelIsFetching)
        return (<div> Loading model {modelName}... </div>);
      return (<div>No available model named {modelName}</div>);
    }
    const width = 800;
    const height = 800;
    const featureWidth = 160;
    const availableData = data[0] || data[1];
    // let modelElement = (<div> Loading Dataset...</div>);
    return (
        <Card>
          {availableData !== undefined &&
          <svg width={featureWidth} height={height}>
            <FeatureList width={featureWidth} featureNames={availableData.featureNames} rules={model.rules} /> 
          </svg>}
          {availableData !== undefined && <Divider type="vertical" />}
          <svg ref={(ref: SVGSVGElement) => this.svgRef = ref} width={width} height={height}>
            <RuleList model={model} data={data} width={width} height={height} /> 
          </svg>
        </Card>
      );
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(ModelView);
