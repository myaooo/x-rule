import * as React from 'react';
import { Action } from 'redux';
// import * as d3 from 'd3';
import { Condition, Rule, Histogram } from '../../models';
import ConditionView from './ConditionView';
import OutputView from './OutputView';
import './index.css';

// const MAX_NUM_RULES = 3;

export interface RuleViewProps {
  rule: Rule;
  featureNames: (i: number) => string;
  labelNames?: (i: number) => string;
  categoryRatios: (feature: number, category: number) => [number, number, number];
  categoryIntervals: (feature: number, category: number) => number | (number | null)[];
  mins: (i: number) => number;
  maxs: (i: number) => number;
  hists?: (i: number) => Histogram;
  fontSize: number;
  transform?: string;
  width: number;
  height: number;
  interval: number;
  nConditions: number;
  selectFeature: ({ idx, deselect }: { idx: number; deselect: boolean }) => Action;
  activatedFeature: number;
  featureIsSelected: boolean;
  logicString: string;
}

export interface RuleViewState {}

export default class RuleView extends React.Component<RuleViewProps, RuleViewState> {
  constructor(props: RuleViewProps) {
    super(props);
    this.state = {};
    this.handleMouseEnter = this.handleMouseEnter.bind(this);
    this.handleMouseLeave = this.handleMouseLeave.bind(this);
  }
  handleMouseEnter(idx: number) {
    this.props.selectFeature({ idx, deselect: false });
  }
  handleMouseLeave(idx: number) {
    this.props.selectFeature({ idx, deselect: true });
  }
  handleClick(idx: number) {
    const { featureIsSelected, activatedFeature, selectFeature } = this.props;
    if (activatedFeature === idx) {
      if (featureIsSelected) {
        selectFeature({ idx, deselect: true });
      } else {
        selectFeature({ idx, deselect: false });
      }
    }
  }
  render() {
    const { rule, featureNames, mins, maxs, categoryIntervals, activatedFeature, hists, categoryRatios } = this.props;
    const { fontSize, transform, width, height, interval, nConditions, logicString, labelNames } = this.props;
    const outputWidth = 120;
    const logicWidth = 60;
    const conditionWidth = (width - outputWidth - logicWidth - (nConditions - 1) * interval) / nConditions;
    const isDefaultRule = rule.conditions[0].feature === -1;
    const outputView = (
      <OutputView
        output={rule.output}
        width={outputWidth}
        height={height}
        barWidth={10}
        interval={10}
        transform={`translate(${width - outputWidth + 30},${0})`}
        labels={labelNames}
      />
    );

    return (
      <g transform={transform}>
        <text textAnchor="end" x={logicWidth - 4} y={height / 2 + 5}>
          {logicString}
        </text>
        {!isDefaultRule &&
          rule.conditions.map((condition: Condition, i: number) => {
            const { feature, category } = condition;
            const featureName = featureNames(feature);
            const categoryInterval = categoryIntervals(feature, category);
            // const min = condition.feature
            return (
              <ConditionView
                key={i}
                onMouseEnter={e => this.handleMouseEnter(feature)}
                onMouseLeave={e => this.handleMouseLeave(feature)}
                onClick={e => this.handleClick(feature)}
                featureName={featureName}
                category={categoryInterval}
                width={conditionWidth}
                height={height}
                min={mins(condition.feature)}
                max={maxs(condition.feature)}
                hist={hists ? hists(condition.feature) : undefined}
                transform={`translate(${(conditionWidth + interval) * i + logicWidth},${0})`}
                fontSize={fontSize}
                activated={activatedFeature === feature}
                ratios={categoryRatios(feature, category)}
              />
            );
          })}
        {!isDefaultRule &&
          rule.conditions.slice(0, -1).map((condition: Condition, i: number) => {
            const pathData = `M ${
              conditionWidth * (i + 1) + interval * (i + 0.5) + logicWidth + (hists ? 0 : 10)
            } ${10} v ${height - 20}`;
            return (
              <path key={i} d={pathData} stroke="#aaa" strokeWidth="0.5" />
            );
          })}
        {outputView}
      </g>
    );
  }
}
