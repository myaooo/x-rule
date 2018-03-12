import * as React from 'react';
// import * as d3 from 'd3';
import { Card } from 'antd';
import { connect } from 'react-redux';
import {
  Dispatch,
  fetchModelIfNeeded,
  getModel,
  RootState,
  getSelectedData,
  getModelIsFetching,
  TreeStyles,
  getTreeStyles
} from '../store';
import { RuleModel, DataSet, ModelBase, isRuleModel, isTreeModel, ConditionalStreams, Streams } from '../models';
import { countFeatureFreq } from '../service/utils';
import Tree from '../components/Tree';
import RuleList from '../containers/RuleList';
import FeatureList from '../containers/FeatureList';
import Legend from '../components/Legend';
import { getStreams, getRuleStyles } from '../store/selectors';
import { Patterns } from '../components/patterns';
import { RuleStyles } from '../store/state';

export interface ModelViewStateProp {
  model: RuleModel | ModelBase | null;
  modelIsFetching: boolean;
  data: (DataSet)[];
  streams?: Streams | ConditionalStreams;
  treeStyles: TreeStyles;
  ruleStyles: RuleStyles;
}

const mapStateToProps = (state: RootState): ModelViewStateProp => {
  return {
    model: getModel(state),
    modelIsFetching: getModelIsFetching(state),
    data: getSelectedData(state),
    treeStyles: getTreeStyles(state),
    ruleStyles: getRuleStyles(state),
    streams: getStreams(state),
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
  componentWillReceiveProps(nextProps: ModelViewProp) {
    const { loadModel, modelName } = nextProps;
    if (modelName !== this.props.modelName)
      loadModel(modelName);
  }
  // componentDidUpdate() {
  //   const { width, height } = {width: 960, height: 720};
  //   d3.select(this.svgRef).attr('width', width).attr('height', height);
  // }
  render(): React.ReactNode {
    const { modelIsFetching, model, data, modelName, treeStyles, ruleStyles, streams } = this.props;
    if (model === null) {
      if (modelIsFetching)
        return (<div> Loading model {modelName}... </div>);
      return (<div>Cannot load model {modelName}</div>);
    }
    const width = 1200;
    const height = 2500;
    const featureWidth = 160;
    // const availableData = data[0] || data[1];
    const transform = `translate(${featureWidth}, 40)`;
    const modelProps = {data, width: width - featureWidth - 20, height: height - 60, transform, streams};
    const featureNames = model.meta.featureNames;
    const labelNames = model.meta.labelNames;
    // let modelElement = (<div> Loading Dataset...</div>);
    // console.log(streams); // tslint:disable-line
    return (
      <Card>
        <svg ref={(ref: SVGSVGElement) => this.svgRef = ref} width={width} height={height}>
        <Patterns />
          { (isTreeModel(model) || ruleStyles.mode === 'list') &&
          <FeatureList 
              width={featureWidth} 
              featureNames={featureNames} 
              featureCounts={countFeatureFreq(model, featureNames.length)}
              transform={`translate(5, 5)`}
          />
          }
          {isRuleModel(model) && 
            <RuleList {...modelProps} model={model} />
          }
          {isTreeModel(model) && 
            <Tree {...modelProps} model={model} styles={treeStyles} />
          }
          <Legend labels={labelNames} transform={`translate(${featureWidth + 20}, 5)`}/>
          
        </svg>
      </Card>
    );
  }
}

export default connect(mapStateToProps, mapDispatchToProps)(ModelView);
