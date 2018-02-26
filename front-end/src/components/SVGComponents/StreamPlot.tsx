import * as React from 'react';
import * as d3 from 'd3';

// import * as nt from '../../service/num';
import { ColorType, labelColor } from '../Painters/Painter';
import { Stream } from '../../models';

type Section = number[] | Int32Array;
// type Stream = Section[];

function process(stream: Stream): Section[] {
  const ret = new Array(stream[0].length);
  for (let i = 0; i < ret.length; ++i) {
    ret[i] = stream.map((s) => s[i]);
  }
  return ret;
}

interface OptionalProps {
  width: number;
  height: number;
  // min: number;
  // max: number;
  color: ColorType;
  transform?: string;
  display: boolean;
  margin: {top: number, bottom: number, left: number, right: number};
  // onMouseEnter?: React.MouseEventHandler<SVGGElement>;
  // onMouseLeave?: React.MouseEventHandler<SVGGElement>;
  // onClick?: React.MouseEventHandler<SVGGElement>;
}

export interface StreamPlotProps extends Partial<OptionalProps> {
  data: Stream;
  range?: [number, number];
  onClick?: () => void;
  className?: string;
}

export interface StreamPlotState {
}

export default class StreamPlot extends React.PureComponent<StreamPlotProps, StreamPlotState> {
  public static defaultProps: OptionalProps = {
    width: 100,
    height: 50,
    color: labelColor,
    display: true,
    margin: {top: 5, bottom: 5, left: 5, right: 5},
  };
  constructor(props: StreamPlotProps) {
    super(props);
  }
  
  render() {
    const { data, range, width, height, color, display, margin, ...rest } 
      = this.props as StreamPlotProps & OptionalProps;

    const chartWidth = width - margin.left + margin.right;
    // const chartHeight = height - margin.top + margin.bottom;
    const processed = process(data);
    const stack = d3.stack<Section, number>().keys(d3.range(data.length)).offset(d3.stackOffsetWiggle);
    const streams = stack(processed);
    const yMin = d3.min(streams, (stream) => d3.min(stream, (d) => d[0])) || -100;
    const yMax = d3.max(streams, (stream) => d3.max(stream, (d) => d[1])) || 100;
    
    const xScaler = d3.scaleLinear()
      .domain([0, processed.length - 1]).range([margin.left, width - margin.right]);
    const yScaler = d3.scaleLinear()
      .domain([yMin, yMax]).range([margin.bottom, height - margin.top]);
    
    const area = d3.area<d3.SeriesPoint<number[]>>()
      .x((d, i) => xScaler(i))
      .y0((d, i) => yScaler(d[0]))
      .y1((d, i) => yScaler(d[1]))
      .curve(d3.curveCardinal.tension(0.3));
    console.log('streams'); //tslint:disable-line
    console.log(streams); //tslint:disable-line
    console.log(processed); //tslint:disable-line
    return (
      <g style={{display}} {...rest}>
        {streams.map((stream: d3.Series<number[], number>, i: number) => (
          <path key={i} d={area(stream) || undefined} fill={color(i)}/>
        ))}
        {range && 
        <rect 
          x={margin.left + chartWidth * range[0]}
          width={chartWidth * (range[1] - range[0])} 
          height={height} 
          style={{fillOpacity: 0.1}}
        />}
      </g>
    );
  }
}
