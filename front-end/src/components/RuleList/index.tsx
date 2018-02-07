import * as React from 'react';
import { Action } from 'redux';
// import * as d3 from 'd3';
import { Rule, RuleModel, PlainData } from '../../models';
import './index.css';
import RuleView from './RuleView';

// interface ExtendRule extends Condition {
//   model: RuleModel;
// }

/*
interface RuleStyle {
  fontSize: number;
  width: number;
  height: number;
  margin: number;
}

function renderRule(
  selector: d3.Selection<BaseType, Rule, Element | null, {}>,
  { discretizers }: RuleModel,
  featureNames: (i: number) => string,
  { fontSize, width, height, margin }: RuleStyle
): void {
  // const discretizer = discretizers[]
  // const intervals = discretizers[].intervals;
  // const mins = discretizer.mins;
  // const maxs = discretizer.maxs;
  const conditionWidth = (width - 2 * margin) / 3;
  const textHeight = fontSize * 1.5;
  const conditionGroup = selector
    .selectAll('g')
    .data((d: Rule) => d.conditions)
    .enter()
    .append('g')
    .attr('transform', (c: Condition, i: number, ): string => `translate(${(conditionWidth + margin) * i}, 0)`);
  // const bgRect = 
  conditionGroup.append('rect')
    .attr('y', height - textHeight)
    .attr('width', conditionWidth)
    .attr('height', textHeight)
    .classed('bg-rect', true);
  // const rangeRect = 
  conditionGroup.append('rect')
    .attr('y', height - textHeight)
    .attr('width', (c: Condition): number => {
      // The last (default) rule
      if (c.feature === -1) return conditionWidth;
      const discretizer = discretizers[c.feature];
      if (discretizer === null) return conditionWidth;
      const interval = discretizer.intervals[c.category];
      const range = (interval[1] === null ? discretizer.max : interval[1])
        - (interval[0] === null ? discretizer.min : interval[0]);
      return conditionWidth * (range / (discretizer.max - discretizer.min));
    })
    .attr('height', textHeight)
    .classed('range-rect', true);
  // const featureText = 
  conditionGroup.append('text')
    .attr('text-anchor', 'middle')
    .attr('x', conditionWidth / 2)
    .attr('y', height - fontSize * 0.25)
    .text((c: Condition): string => featureNames(c.feature));
} */

export interface RuleListProps {
  // rules: Rule[];
  model: RuleModel;
  data?: PlainData;
  fontSize?: number;
  width: number;
  height: number;
  margin?: number;
  interval?: number;
  selectFeature: ({idx, deselect}: {idx: number, deselect: boolean}) => Action;
  activatedFeature: number;
  featureIsSelected: boolean;
}

class RuleListView extends React.Component<RuleListProps, {}> {
  ref: SVGGElement | null;
  margin: number;
  interval: number;
  fontSize: number;
  constructor(props: RuleListProps) {
    super(props);
    this.ref = null;
    this.margin = props.margin || 5;
    this.interval = props.interval || 10;
    this.fontSize = props.fontSize || 11;
  }
  update(rules: Rule[]) {
    // const { data, model } = this.props;
    // const { fontSize, interval, margin, ref } = this;
    // const nRules = rules.length;
    // const width = this.props.width - margin * 2;
    // const height = this.props.height - margin * 2;
    // const ruleHeight = (height - interval * (nRules - 1)) / rules.length;
    // const rootGroup = d3.select(ref).attr('transform', `translate(${margin}, ${margin})`);
    // const ruleGroup = rootGroup
    //   .selectAll('g')
    //   .data(rules)
    //   .enter()
    //   .append('g')
    //   .attr('transform', (rule: Rule, i: number) => `translate(0,${(ruleHeight + interval) * i})`);
    // let featureNames = ((i: number): string => `X${i}`);
    // if ((data) !== undefined)
    //   featureNames = ((i: number): string => data.featureNames[i]);
    // renderRule(ruleGroup, model, featureNames, { fontSize, width, height: ruleHeight, margin });
  }
  componentDidMount() {
    this.update(this.props.model.rules);
  }
  componentWillUpdate(nextProps: RuleListProps) {
    if (nextProps.data !== undefined) {
      console.log(nextProps.data); // tslint:disable-line
      this.update(this.props.model.rules);
    }
  }
  render() {
    const { model, data, width, height, selectFeature, activatedFeature, featureIsSelected } = this.props;
    const { fontSize, interval, margin } = this;
    
    const nRules = model.rules.length;
    let ruleHeight = (height - 2 * margin - (nRules - 1) * interval) / nRules;
    ruleHeight = Math.max(80, Math.min(ruleHeight, 100));
    const discretizers = model.discretizers;
    let featureNames = ((i: number): string => `X${i}`);
    let labelNames = ((i: number): string => `L${i}`);
    if ((data) !== undefined) {
      featureNames = ((i: number): string => data.featureNames[i]);
      labelNames = ((i: number): string => data.labelNames[i]);
    }
    console.log('discretizers');  //tslint:disable-line
    console.log(discretizers);  //tslint:disable-line
    const categoryIntervals = (feature: number, cat: number) => {
      if (feature === -1) return 0;
      console.log(`discretizers ${feature} ${cat}`);  //tslint:disable-line
      console.log(model);  //tslint:disable-line
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
    return (
      <g ref={(ref: SVGGElement) => (this.ref = ref)} transform={`translate(${margin}, ${margin})`}>
        {model.rules.map((rule: Rule, i: number) => {
          return (
            <g key={i}>
              <RuleView
                logicString={i === 0 ? 'IF' : ((i === nRules - 1) ? 'DEFAULT' : 'ELSE IF')}
                rule={rule} 
                width={width - 2 * margin} 
                height={ruleHeight}
                interval={interval}
                featureNames={featureNames}
                labelNames={labelNames}
                categoryIntervals={categoryIntervals}
                mins={(j: number) => discretizers[j].min}
                maxs={(j: number) => discretizers[j].max}
                hists={data ? ((j: number) => data.hists[j]) : undefined}
                fontSize={fontSize}
                transform={`translate(0,${(ruleHeight + interval) * i})`}
                nConditions={nConditions}
                selectFeature={selectFeature}
                activatedFeature={activatedFeature}
                featureIsSelected={featureIsSelected}
                categoryRatios={categoryRatios}
              />
              <path 
                d={`M 10 ${(ruleHeight + interval) * (i + 1) - interval / 2} H ${width - 2 * margin - 20}`} 
                stroke="#555"
                strokeWidth="1px"
              />
            </g>
          );
        })}
      </g>
    );
  }
}

export default RuleListView;
