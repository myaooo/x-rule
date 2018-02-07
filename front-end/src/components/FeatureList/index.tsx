import * as React from 'react';
import { Action } from 'redux';
import { Rule } from '../../models';
import './index.css';

export interface FeatureProps {
  x: number;
  y: number;
  unitLength: number;
  nRefs: number;
  fontSize: number;
  featureName: string;
  isSelected: boolean;
  onMouseEnter: React.MouseEventHandler<SVGGElement>;
  onMouseLeave: React.MouseEventHandler<SVGGElement>;
  onClick: React.MouseEventHandler<SVGGElement>;
}

export interface FeatureState {
  // isSelected: boolean;
}

export class Feature extends React.Component<FeatureProps, FeatureState> {
  constructor(props: FeatureProps) {
    super(props);
    // this.state = {
    //   isSelected: false
    // };
  }

  render() {
    const { featureName, unitLength, fontSize, x, y, nRefs} = this.props;
    const { onMouseEnter, onMouseLeave, onClick, isSelected } = this.props;
    const height = fontSize * 1.5;
    const yText = height - fontSize * 0.3;
    const width = unitLength * nRefs;
    return (
      <g transform={`translate(${x},${y})`} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} onClick={onClick}>
        <rect width={width} height={height} className={isSelected ? 'feature-rect-active' : 'feature-rect'}/>
        <text x={2} y={yText} textAnchor="right" fontSize={fontSize} fontWeight={isSelected ? 700 : 300}>
          {featureName}  ({nRefs})</text>
        {/* <text x={width + 2} y={height - fontSize * 0.3} textAnchor="right" fontSize={fontSize}>{nRefs}</text> */}
      </g>
    );
  }
}

export interface FeatureListProps {
  width: number;
  interval?: number;
  fontSize?: number;
  margin?: number;
  featureNames: string[];
  rules: Rule[];
  selectFeature?: ({idx, deselect}: {idx: number, deselect: boolean}) => Action;
  activatedFeature?: number;
  featureIsSelected?: boolean;
}

export interface FeatureListState {
}

export default class FeatureList extends React.Component<FeatureListProps, FeatureListState> {
  margin: number;
  interval: number;
  fontSize: number;
  constructor(props: FeatureListProps) {
    super(props);
    this.margin = props.margin || 5;
    this.interval = props.interval || 5;
    this.fontSize = props.fontSize || 11;
    this.state = {};
  }
  handleMouseEnter(idx: number) {
    const {selectFeature} = this.props;
    if (selectFeature)
      selectFeature({ idx, deselect: false });
  }
  handleMouseLeave(idx: number) {
    const {selectFeature} = this.props;
    if (selectFeature)
      selectFeature({ idx, deselect: true });
  }
  handleClick(idx: number) {
    const { featureIsSelected, activatedFeature, selectFeature } = this.props;
    if (selectFeature) {
      if (activatedFeature === idx) {
        if (featureIsSelected) {
          selectFeature({ idx, deselect: true });
        } else {
          selectFeature({ idx, deselect: false });
        }
      }
    }
  }
  render() {
    const {width, featureNames, rules, activatedFeature} = this.props;
    const { margin, interval, fontSize } = this;
    const counts = new Array(featureNames.length).fill(0);
    rules.forEach(rule => {
      rule.conditions.forEach(condition => {
        if (condition.feature !== -1)
          counts[condition.feature] ++;
      });
    });
    const features = featureNames.map((featureName: string, i: number) => ({
      featureName,
      count: counts[i],
      idx: i
    }));
    features.sort((a, b) => b.count - a.count);
    const maxCount = Math.max(...counts);
    const unitLength = (width - 2 * margin - 20) / maxCount;
    return (
      <g transform={`translate(${margin},${margin})`}>
        {features.map(({featureName, count, idx}: {featureName: string, count: number, idx: number}, i: number) => {
          return (
            <Feature 
              key={idx}
              onMouseEnter={e => this.handleMouseEnter(idx)}
              onMouseLeave={e => this.handleMouseLeave(idx)}
              onClick={e => this.handleClick(idx)}
              x={0} 
              y={(interval + fontSize * 1.5) * i} 
              unitLength={unitLength} 
              nRefs={count} 
              featureName={featureName} 
              fontSize={fontSize}
              isSelected={activatedFeature === idx}
            />
          );
        })}
      </g>
    );
  }
}
