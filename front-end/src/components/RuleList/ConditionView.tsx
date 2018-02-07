import * as React from 'react';
import * as d3 from 'd3';
import './index.css';

import { Histogram } from '../../models';

// const MAX_NUM_RULES = 3;

export interface ConditionViewProps {
  featureName: string;
  category: (number | null)[] | number;
  hist?: Histogram;
  width: number;
  height: number;
  min: number;
  max: number;
  ratios: [number, number, number];
  transform: string;
  fontSize: number;
  activated: boolean;
  onMouseEnter: React.MouseEventHandler<SVGGElement>;
  onMouseLeave: React.MouseEventHandler<SVGGElement>;
  onClick: React.MouseEventHandler<SVGGElement>;
  // selectFeature: ({deselect}: {deselect: boolean}) => Action;
}

export interface ConditionViewState {
  // activated: boolean;
}

function condition2String(featureName: string, category: (number | null)[] | number): string {
  let conditionString: string;
  if (typeof category === 'number') {
    conditionString = `${featureName} = ${category}`;
  } else {
    if (category[0] === null && category[1] === null) conditionString = `${featureName} is any`;
    else {
      conditionString = `${featureName}`;
      if (category[0] !== null) conditionString = `${category[0]} < ` + conditionString;
      if (category[1] !== null) conditionString = conditionString + ` < ${category[1]}`;
    }
  }
  return conditionString;
}

function interval2Range(interval: (number | null)[], min: number, max: number): { low: number; high: number } {
  let low = (interval[0] === null ? min : interval[0]) as number;
  const high = (interval[1] === null ? max : interval[1]) as number;
  return { high, low };
}

// type Range = [number, number];

export default class ConditionView extends React.Component<ConditionViewProps, ConditionViewState> {
  // refCounts: number;
  xAxisRef: SVGGElement;
  yAxisRef: SVGGElement;
  xScale: d3.ScaleLinear<number, number>;
  yScale: d3.ScaleLinear<number, number>;
  margin: { top: number; bottom: number; left: number; right: number };
  
  constructor(props: ConditionViewProps) {
    super(props);
    this.state = { activated: false };
    this.margin = { top: 10, bottom: 35, left: 25, right: 5 };
  }

  renderHist(hist: Histogram) {
    const margin = this.margin;
    const { width, height, category, min, max } = this.props;
    // const interval = hist.centers[1] - hist.centers[0];
    // const nBins = hist.centers.length;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    let lineData: [number, number][] = hist.counts.map((count: number, i: number): [number, number] => {
      return [hist.centers[i], count];
    });
    // this.min = hist.centers[0] - interval / 2;
    // this.max = hist.centers[nBins - 1] + interval / 2;
    lineData = [[min, 0], ...lineData, [max, 0]];
    const xScale = d3
      .scaleLinear()
      .domain([min, max]) // input
      .range([1, chartWidth - 1]); // output

    const yScale = d3
      .scaleLinear()
      .domain([0, Math.max(...hist.counts)]) // input
      .range([chartHeight, 0]); // output
    
    this.xScale = xScale;
    this.yScale = yScale;

    const line = d3
      .line<[number, number]>()
      .x((d: number[]): number => xScale(d[0]))
      .y((d: number[]): number => yScale(d[1]))
      .curve(d3.curveCardinal.tension(0.4));
    const lineString = line(lineData);
    const area = d3
      .area<[number, number]>()
      .x((d: number[]): number => xScale(d[0]))
      .y1((d: number[]): number => yScale(d[1]))
      .y0(yScale(0))
      .curve(d3.curveCardinal.tension(0.4));
    const areaString = area(lineData);

    // if (lineString === null || areaString === null) return '';
    let highLightedArea = null;
    // let highLightedLineString = null;
    if (typeof category !== 'number') {
      const { low, high } = interval2Range(category, min, max);
      highLightedArea = (
        <g>
          <path d={`M ${xScale(low)} 0 v ${chartHeight}`} className="feature-dist-highlight"/>
          <path d={`M ${xScale(high)} 0 v ${chartHeight}`} className="feature-dist-highlight"/>
        <rect 
          x={xScale(low)} 
          y={0} 
          width={xScale(high) - xScale(low)} 
          height={chartHeight} 
          className="feature-dist-area-highlight"
        />
        </g>
      );
    }
    return (
      <g transform={`translate(${margin.left},${margin.top})`}>
        {areaString && <path className="feature-dist-area" d={areaString} />}
        {lineString && <path className="feature-dist" d={lineString} />}
        {/* {highLightedAreaString && <path className="feature-dist-area-highlight" d={highLightedAreaString} />} */}
        {highLightedArea}
      </g>
    );

  }
  renderHistAxis() {
    d3
      .select(this.xAxisRef)
      .attr('class', 'x-axis')
      .attr('transform', `translate(${this.margin.left},${this.props.height - this.margin.bottom})`)
      .call(d3.axisBottom(this.xScale).ticks(5).tickSize(3));
    d3
      .select(this.yAxisRef)
      .attr('class', 'y-axis')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`)
      .call(d3.axisLeft(this.yScale).ticks(2).tickSize(3));
  }
  renderCondition() {
    const { featureName, category, width, height, fontSize, activated, hist, ratios } = this.props;
    const { margin } = this;
    const conditionHeight = Math.ceil(fontSize * 1.5);
    const conditionWidth = width - margin.left - margin.right;
    const conditionString = condition2String(featureName, category);
    let rangeWidth: number = conditionWidth * ratios[1];
    let rangeX: number = conditionWidth * ratios[0];
    // if (typeof category !== 'number') {
    //   const { low, high } = interval2Range(category, min, max);
    //   rangeWidth = (high - low) / (max - min) * conditionWidth;
    //   rangeX += (low - min) / (max - min) * conditionWidth;
    // }
    return (
      <g transform={`translate(${margin.left},${hist ? height : ((height + conditionHeight) / 2)})`}>
        <rect
          y={-conditionHeight}
          width={conditionWidth}
          height={5}
          className={activated ? 'bg-rect-active' : 'bg-rect'}
        />
        <rect
          x={rangeX + 1}
          y={-conditionHeight + 1}
          width={rangeWidth - 2}
          height={5 - 2}
          className="range-rect"
        />
        <text x={conditionWidth / 2} y={-fontSize * 0.2} textAnchor="middle" fontSize={fontSize}>
          {conditionString}
        </text>
      </g>
    );
  }
  componentDidUpdate() {
    // Hack: make sure the element is already mounted, then render the axis
    if (this.props.hist !== undefined) this.renderHistAxis();
  }
  render() {
    const { transform, hist } = this.props;
    const { onMouseEnter, onMouseLeave, onClick } = this.props;
    const conditionElement = this.renderCondition();
    const histElement = hist === undefined ? '' : this.renderHist(hist);
    return (
      <g transform={transform} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} onClick={onClick}>
        {conditionElement}
        {histElement}
        <g ref={(ref: SVGGElement) => (this.xAxisRef = ref)} />
        <g ref={(ref: SVGGElement) => (this.yAxisRef = ref)} />
      </g>
    );
  }
}
