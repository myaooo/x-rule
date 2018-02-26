import * as React from 'react';
import { Action } from 'redux';
import * as d3 from 'd3';
import { Rule, RuleModel, DataSet } from '../../models';
import './index.css';
import RuleView from './RuleView';
import { collapsedHeight, expandedHeight } from './ConditionView';
import { FeatureStatus, RuleStyles } from '../../store';
import { RuleListPainter } from './Painter';
import RuleMatrix from '../RuleMatrix';

export interface RuleListProps {
  // rules: Rule[];
  model: RuleModel;
  data?: DataSet[];
  fontSize?: number;
  width: number;
  height: number;
  transform?: string;
  interval?: number;
  styles?: RuleStyles;
  // mode?: 'list' | 'matrix';
  selectFeature({idx, deselect}: {idx: number, deselect: boolean}): Action;
  featureStatus(idx: number): FeatureStatus;
}

export interface RuleListState {
  unCollapsedSet: Set<number>;
  // interval: number;
  // fontSize: number;
}

class RuleListView extends React.Component<RuleListProps, RuleListState> {
  public static defaultProps: Partial<RuleListProps> = {
    interval: 15,
    // mode: 'matrix',
  };
  private ref: SVGGElement;
  private painter: RuleListPainter;
  // private refreshFlag: boolean;
  constructor(props: RuleListProps) {
    super(props);
    // this.refreshFlag = false;
    this.painter = new RuleListPainter().data(props.model);
    // this.handleClickCollapse = this.handleClickCollapse.bind(this);
  }
  handleClickCollapse(idx: number, collapsed: boolean) {
    const unCollapsedSet = new Set(this.state.unCollapsedSet);
    if (collapsed) unCollapsedSet.delete(idx);
    else unCollapsedSet.add(idx);
    this.setState({unCollapsedSet});
  }
  update() {
    const { data, model, styles } = this.props;
    if (styles === undefined || styles.mode === 'list') {
      const selector = d3.select(this.ref);
      this.painter.data(model).update({data}).render(selector);
      // this.refreshFlag = false;
    }
  }
  componentDidMount() {
    this.update();
  }
  shouldComponentUpdate(nextProps: RuleListProps, nextState: RuleListState) {
    return true;
    // return this.refreshFlag;
  }
  // componentWillReceiveProps(nextProps: RuleListProps) {
  //   this.refreshFlag = true;
  // }
  componentDidUpdate() {
    this.update();
  }
  render() {
    const {transform, styles, model, data} = this.props;
    if (styles === undefined || styles.mode === 'list') {
      return <g ref={(ref: SVGGElement) => (this.ref = ref)} transform={transform}/>;
    }
    return <RuleMatrix model={model} datasets={data || []} transform={transform} {...styles}/>;
  }
  renderBackUp() {
    const { model, data, width, selectFeature, transform, featureStatus, interval } = this.props;

    const { unCollapsedSet } = this.state;
    
    const nRules = model.rules.length;
    // const ruleHeight = Math.max(80, Math.min((height - 2 * margin - (nRules - 1) * interval) / nRules, 100));
    const discretizers = model.discretizers;
    let featureNames = ((i: number): string => `X${i}`);
    let labelNames = ((i: number): string => `L${i}`);
    if (data && data.length) {
      featureNames = ((i: number): string => data[0].featureNames[i]);
      labelNames = ((i: number): string => data[0].labelNames[i]);
    }

    const categoryIntervals = (feature: number, cat: number) => {
      if (feature === -1) return 0;
      const intervals = discretizers[feature].intervals;
      if (intervals === null) return cat;
      return intervals[cat];
    };

    const categoryRatios = (feature: number, cat: number): [number, number, number] => {
      const ratios = discretizers[feature].ratios;
      let prevSum = 0;
      for (let i = 0; i < cat; i++) {
        prevSum += ratios[i];
      }
      return [prevSum, ratios[cat], 1 - prevSum - ratios[cat]];
    };

    const nConditions = Math.max(...(model.rules.map(rule => rule.conditions.length)));
    const passDown = {
      selectFeature, featureStatus, categoryRatios, nConditions, featureNames, labelNames, categoryIntervals
    };
    let heightSum = 0;

    return (
      <g ref={(ref: SVGGElement) => (this.ref = ref)} transform={transform}>
        {model.rules.map((rule: Rule, i: number) => {
          const collapsed = !unCollapsedSet.has(i);
          const yTransform = heightSum;
          heightSum += (collapsed ? collapsedHeight : expandedHeight) + (interval as number);
          return (
            <g key={i}>
              <RuleView
                logicString={i === 0 ? 'IF' : ((i === nRules - 1) ? 'DEFAULT' : 'ELSE IF')}
                rule={rule} 
                support={model.supports[i]}
                width={width} 
                interval={interval as number}
                mins={(j: number) => discretizers[j].min}
                maxs={(j: number) => discretizers[j].max}
                hists={(data && data.length) ? ((j: number) => data.map(d => d.hists[j])) : undefined}
                transform={`translate(0,${yTransform})`}
                collapsed={collapsed}
                onClickCollapse={(c: boolean) => this.handleClickCollapse(i, c)}
                {...passDown}
              />
            </g>
          );
        })}
      </g>
    );
  }
}

export default RuleListView;
