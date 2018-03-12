import * as React from 'react';
import * as d3 from 'd3';

import * as nt from '../../service/num';
import { ColorType, labelColor } from '../Painters/Painter';
// import { Stream } from '../../models';
import './Histogram.css';

type Hist = number[] | Int32Array;

function checkBins (hists: Hist[]) {
  let nBins = hists[0].length;
  let equalBins: boolean = true;
  for (let i = 0; i < hists.length; ++i) {
    if (nBins !== hists[i].length)
      equalBins = false;
  }
  if (!equalBins) {
    console.warn('Hists not having the same number of bins!');
    hists.forEach((h) => nBins = Math.max(nBins, h.length));
  }
  return {nBins, equalBins};
}

interface OptionalProps {
  width: number;
  height: number;
  // min: number;
  // max: number;
  mode: 'stack' | 'overlay';
  color: ColorType;
  transform?: string;
  display: boolean;
  margin: {top: number, bottom: number, left: number, right: number};
  // onMouseEnter?: React.MouseEventHandler<SVGGElement>;
  // onMouseLeave?: React.MouseEventHandler<SVGGElement>;
  // onClick?: React.MouseEventHandler<SVGGElement>;
}

export interface HistogramProps extends Partial<OptionalProps> {
  hists: Hist[];
  range?: [number, number];
  className?: string;
}

export interface HistogramState {
}

export default class Histogram extends React.PureComponent<HistogramProps, HistogramState> {
  public static defaultProps: OptionalProps = {
    width: 100,
    height: 50,
    color: labelColor,
    display: true,
    margin: {top: 5, bottom: 5, left: 5, right: 5},
    mode: 'overlay'
  };
  constructor(props: HistogramProps) {
    super(props);
  }

  renderOverlay() {
    const { hists, range, width, height, color, display, margin, mode, ...rest } 
      = this.props as HistogramProps & OptionalProps;
    if (hists.length === 0) return;
    const {nBins} = checkBins(hists);
    const yMax = d3.max(hists, (hist) => d3.max(hist)) as number;
    const chartWidth = width - margin.left - margin.right;
    const padding = Math.min(chartWidth / (2 * nBins), 5);
    const xScaler = d3.scaleBand<number>()
      .domain([0, nBins]).range([margin.left, width - margin.right])
      .padding(padding);
    const yScaler = d3.scaleLinear().domain([yMax, 0]).range([margin.bottom, height - margin.top]);
    const hScaler = d3.scaleLinear().domain([0, yMax]).range([0, height - margin.top - margin.bottom]);
    const bandWidth = xScaler.bandwidth();
    const r0 = range ? range[0] : 0;
    const r1 = range ? range[1] : nBins;
    const indices = d3.range(nBins);
    return (
      <g style={{display}} {...rest}>
        {hists.map((hist: Hist, i: number) => {
          return (
            <React.Fragment key={i}>
              {indices.map((j) => {
                const h = hist[j];
                return (
                  <rect 
                    key={j} 
                    x={xScaler(j)} 
                    y={yScaler(h)} 
                    width={bandWidth} 
                    height={hScaler(h)} 
                    fill={color(i)}
                    className={(range && r0 < j && j < r1 ) ? 'hist-active' : 'hist'}
                  />
                );
              })}
            </React.Fragment>
          );
        })}
        {range && 
        <rect 
          x={xScaler(range[0]) || margin.left}
          width={(xScaler(range[1]) || (width - margin.right)) - (xScaler(range[0]) || margin.left)} 
          height={height} 
          className="hist-brush"
        />}
      </g>
    );
  }

  renderStack() {
    const { hists, range, width, height, color, display, margin, mode, ...rest } 
      = this.props as HistogramProps & OptionalProps;
    if (hists.length === 0) return;
    const {nBins, equalBins} = checkBins(hists);
    if (!equalBins) return;
    const y1s = nt.stack(hists);
    const y0s = [new Array(nBins).fill(0), ...(y1s.slice(0, -1))];
    const yMax = d3.max(y1s[y1s.length - 1]) as number;
    const chartWidth = width - margin.left - margin.right;
    const padding = Math.min(chartWidth / (2 * nBins), 5);
    const xScaler = d3.scaleBand<number>()
      .domain([0, nBins]).range([margin.left, width - margin.right])
      .padding(padding);
    const yScaler = d3.scaleLinear().domain([yMax, 0]).range([margin.bottom, height - margin.top]);
    const hScaler = d3.scaleLinear().domain([0, yMax]).range([0, height - margin.top - margin.bottom]);
    const bandWidth = xScaler.bandwidth();

    const r0 = range ? range[0] : 0;
    const r1 = range ? range[1] : nBins;
    const indices = d3.range(nBins);
    return (
      <g style={{display}} {...rest}>
        {hists.map((hist: Hist, i: number) => {
          return (
            <React.Fragment key={i}>
              {indices.map((j) => {
                return (
                  <rect 
                    key={j} 
                    x={xScaler(j)} 
                    y={yScaler(y0s[i][j])} 
                    width={bandWidth} 
                    height={hScaler(hist[j])} 
                    fill={color(i)}
                    className={(range && r0 < j && j < r1 ) ? 'hist-active' : 'hist'}
                  />
                );
              })}
            </React.Fragment>
          );
        })}
        {range && 
        <rect 
          x={xScaler(range[0]) || margin.left}
          width={(xScaler(range[1]) || (width - margin.right)) - (xScaler(range[0]) || margin.left)} 
          height={height} 
          className="hist-brush"
        />}
      </g>
    );
  }
  render() {
    const mode = this.props.mode;
    if (mode === 'stack') return this.renderStack();
    return this.renderOverlay();
  }
}
