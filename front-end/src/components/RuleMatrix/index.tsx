import * as React from 'react';
import { NodeGroup } from 'react-move';

import { RuleList, Rule, Condition, DataSet, Streams, ConditionalStreams, isConditionalStreams } from '../../models';
import * as nt from '../../service/num';
import TextGroup from '../SVGComponents/TextGroup';
import VerticalFlow from '../SVGComponents/VerticalFlow';
import RuleRow from './RuleRow';
import RowOutput from './RowOutput';

import './index.css';
import { labelColor as defaultLabelColor, ColorType } from '../Painters/Painter';

class RuleMatrixNodeGroup extends NodeGroup<number, {y?: number, height?: number}> {}

interface RuleMatrixPropsOptional {
  transform: string;
  size: number;
  intervalY: number;
  intervalX: number;
  width: number;
  labelColor: ColorType;
}

export interface RuleMatrixProps extends Partial<RuleMatrixPropsOptional> {
  model: RuleList;
  datasets: DataSet[];
  streams?: Streams | ConditionalStreams;
}

export interface RuleMatrixState {
  activeFeatures: Set<number>;
  activeRules: Set<number>;
  features: number[];
  widths: number[];
  heights: number[];
  xs: number[];
  ys: number[];
}

export default class RuleMatrix extends React.Component<RuleMatrixProps, RuleMatrixState> {
  public static defaultProps: Partial<RuleMatrixProps> & RuleMatrixPropsOptional = {
    transform: '',
    size: 40,
    intervalY: 10,
    intervalX: 0.2,
    width: 60,
    labelColor: defaultLabelColor,
  };
  private stateUpdated: boolean;

  public static computeSizes(newSize: number) {
    const size = newSize || 40;
    return {
      featureWidth: size,
      featureWidthExpand: size * 4,
      ruleHeight: size,
      ruleHeightExpand: size * 2,
    };
  }

  public static computeExistingFeatures(rules: Rule[]) {
    const ruleFeatures = rules.slice(0, -1).map((r: Rule) => r.conditions.map((c: Condition) => c.feature));
    const features = Array.from(new Set(ruleFeatures.reduce((a, b) => a.concat(b)))).sort();
    return features;
  }

  public static computePos(
    props: RuleMatrixProps,
    {activeFeatures, activeRules, features}
    : {activeFeatures: Set<number>, activeRules: Set<number>, features: number[]}
  ) {
    const {model, intervalY, size, intervalX } = props as RuleMatrixPropsOptional & RuleMatrixProps;
    // const {activeFeatures, activeRules, features} = this.state;
    // Get sizes
    const { featureWidth, featureWidthExpand, ruleHeight, ruleHeightExpand } = 
      RuleMatrix.computeSizes(size);

    const rules = model.rules;

    // compute the widths and heights
    const featureWidths = 
      features.map((f: number) => (activeFeatures.has(f) ? featureWidthExpand : featureWidth));
    const ruleHeights = 
      rules.map((r: Rule, i: number) => (activeRules.has(i) ? ruleHeightExpand : ruleHeight));

    let ys = ruleHeights.map((h) => h + intervalY);
    ys = [0, ...(nt.cumsum(ys.slice(0, -1)))];

    let xs = featureWidths.map((w: number) => w + intervalX * size);
    xs = [0, ...(nt.cumsum(xs.slice(0, -1)))];
    // const totalWidth = nt.sum(featureWidths);
    return {
      widths: featureWidths,
      heights: ruleHeights,
      ys, 
      xs
    };
  }
  constructor(props: RuleMatrixProps) {
    super(props);
    this.handleClick.bind(this);
    this.stateUpdated = false;
    const state = {
      features: RuleMatrix.computeExistingFeatures(props.model.rules),
      activeFeatures: new Set(),
      activeRules: new Set(),
    };

    this.state = {...state, ...(RuleMatrix.computePos(props, state))};
  }

  handleClick(r: number, f: number) {
    const { activeFeatures, activeRules, features } = this.state;
    // const activeFeatures = this.state.act;
    // const activeRules = this.state;
    if (activeFeatures.has(f) && activeRules.has(r)) {
      activeRules.delete(r);
      activeFeatures.delete(f);
    } else {
      if (!activeFeatures.has(f)) activeFeatures.add(f);
      if (!activeRules.has(r)) activeRules.add(r);
    }
    const state = RuleMatrix.computePos(this.props, {activeFeatures, activeRules, features});
    // console.log('Clicked'); // tslint:disable-line
    this.stateUpdated = true;
    this.setState({
      activeRules,
      activeFeatures,
      ...state
    });
  }

  componentWillReceiveProps(nextProps: RuleMatrixProps) {
    let state = {};
    if (nextProps.model !== this.props.model) {
      state = {features: RuleMatrix.computeExistingFeatures(nextProps.model.rules)};
      state = {...state, ...(RuleMatrix.computePos(nextProps, this.state))};
      // console.log("Change state due to model change"); //tslint:disable-line
      this.stateUpdated = true;
      this.setState(state);
      return;
    }
    if (nextProps.size && nextProps.size !== this.props.size) {
      // console.log("Change state due to size change"); //tslint:disable-line
      this.stateUpdated = true;
      this.setState(RuleMatrix.computePos(nextProps, this.state));
    }
  }

  shouldComponentUpdate(nextProps: RuleMatrixProps, nextState: RuleMatrixState): boolean {
    const {size, width} = nextProps;
    if (size !== this.props.size || width !== this.props.width) 
      return true;
    if (this.stateUpdated) {
      return true;
    }
    if (nextProps.datasets !== this.props.datasets) {
      return true;
    }
    return false;
  }
  componentDidUpdate() {
    this.stateUpdated = false;
  }

  render() {
    const {model, datasets, transform, width, size, streams, labelColor} 
      = this.props as RuleMatrixPropsOptional & RuleMatrixProps;
    // console.log(rules); // tslint:disable-line
    const getStreams = streams 
      ? (isConditionalStreams(streams) ? ((i: number) => streams[i]) : () => streams) 
      : undefined;
    const {features, widths, heights, xs, ys, activeFeatures} = this.state;
    const rules = model.rules;

    // compute feature2Idx map
    const feature2Idx = new Array(model.nFeatures).fill(-1);
    features.forEach((f, i) => feature2Idx[f] = i);

    // get the existing feature names
    const featureNames = (datasets.length) ? (datasets[0].featureNames) : undefined;
    const selectedFeatureNames = features.map((f: number) => (featureNames ? featureNames[f] : `X${f}`));

    const textXs = xs.map((x, i) => x += widths[i] / 2);
    const midYs = ys.map((y, i) => y + heights[i] / 2);
    const x0 = 150;
    const flowDx = Math.max(50, size + 10);
    const outputX = xs[xs.length - 1] + widths[widths.length - 1] + 10;
    // console.log(xs); // tslint:disable-line
    // const getStreams = streams
    return (
      <g transform={transform}>
        <TextGroup
          texts={[...selectedFeatureNames, 'Confidence']}
          xs={[...textXs, xs[xs.length - 1] + widths[xs.length - 1] + 30]}
          rotate={-45}
          transform={`translate(${x0}, 75)`}
        />
        <RuleMatrixNodeGroup 
          data={heights} 
          keyAccessor={(d, i) => i.toString()}
          start={() => ({y: 0, })}
          enter={(d, i) => ({y: [ys[i]]})}
          update={(d, i) => ({y: [ys[i]]})}
        >
          {(nodes) => {
            // console.log("Render RuleMatrixNodeGroup"); //tslint:disable-line
            return (
            <g transform={`translate(${x0}, 80)`}>
              {nodes.map(({key, data, state}) => {
                const {y} = state;
                const i = Number(key);
                // console.log(rules[i]);  // tslint:disable-line
                return (
                  <g key={key} transform={`translate(0, ${y})`}>
                    <RuleRow 
                      rule={rules[i]} 
                      dataset={datasets[0]}
                      supports={model.supports[i]}
                      features={features}
                      activeFeatures={activeFeatures}
                      feature2Idx={feature2Idx}
                      xs={xs}
                      streams={getStreams && getStreams(i)}
                      widths={widths} 
                      height={heights[i]}
                      onClick={(f) => this.handleClick(Number(key), f)}
                    />
                    <RowOutput
                      outputs={rules[i].output}
                      supports={model.useSupportMat ? model.supportMats[i] : model.supports[i]}
                      height={heights[i]}
                      supportWidth={500}
                      transform={`translate(${outputX}, 0)`}
                      color={labelColor}
                      className="matrix-outputs"
                    />
                  </g>
                );
              })}
            </g>
          );
          }}
        </RuleMatrixNodeGroup>
        <VerticalFlow 
          supports={model.supports} 
          ys={midYs} 
          transform={`translate(${x0 - flowDx}, 80)`} 
          dx={flowDx} 
          dy={size}
          width={width}
        />
      </g>
    );
  }
}
