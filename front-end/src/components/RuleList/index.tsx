import * as React from 'react';
import { Action } from 'redux';
// import * as d3 from 'd3';
import { Rule, RuleModel, PlainData } from '../../models';
import './index.css';
import RuleView from './RuleView';
import { collapsedHeight, expandedHeight } from './ConditionView';
import { FeatureStatus } from '../../store';

export interface RuleListProps {
  // rules: Rule[];
  model: RuleModel;
  data?: PlainData[];
  fontSize?: number;
  width: number;
  height: number;
  transform?: string;
  interval?: number;
  selectFeature({idx, deselect}: {idx: number, deselect: boolean}): Action;
  featureStatus(idx: number): FeatureStatus;
}

export interface RuleListState {
  unCollapsedSet: Set<number>;
  interval: number;
  // fontSize: number;
}

class RuleListView extends React.Component<RuleListProps, RuleListState> {
  ref: SVGGElement | null;
  constructor(props: RuleListProps) {
    super(props);
    this.ref = null;
    this.state = {
      unCollapsedSet: new Set(),
      interval: props.interval || 15,
      // fontSize: props.fontSize || 12,
    };
    this.handleClickCollapse = this.handleClickCollapse.bind(this);
  }
  handleClickCollapse(idx: number, collapsed: boolean) {
    const unCollapsedSet = new Set(this.state.unCollapsedSet);
    if (collapsed) unCollapsedSet.delete(idx);
    else unCollapsedSet.add(idx);
    this.setState({unCollapsedSet});
  }
  render() {
    const { model, data, width, selectFeature, transform, featureStatus } = this.props;
    const { interval, unCollapsedSet } = this.state;
    
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
          heightSum += (collapsed ? collapsedHeight : expandedHeight) + interval;
          return (
            <g key={i}>
              <RuleView
                logicString={i === 0 ? 'IF' : ((i === nRules - 1) ? 'DEFAULT' : 'ELSE IF')}
                rule={rule} 
                width={width} 
                interval={interval}
                mins={(j: number) => discretizers[j].min}
                maxs={(j: number) => discretizers[j].max}
                hists={(data && data.length) ? ((j: number) => data.map(d => d.hists[j])) : undefined}
                transform={`translate(0,${yTransform})`}
                collapsed={collapsed}
                onClickCollapse={(c: boolean) => this.handleClickCollapse(i, c)}
                {...passDown}
              />
              {/* <path 
                d={`M 10 ${(ruleHeight + interval) * (i + 1) - interval / 2} H ${width - 2 * margin - 20}`} 
                stroke="#555"
                strokeWidth="1px"
              /> */}
            </g>
          );
        })}
      </g>
    );
  }
}

export default RuleListView;
