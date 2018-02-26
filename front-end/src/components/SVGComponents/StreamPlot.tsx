import * as React from 'react';
import * as d3 from 'd3';

// import * as nt from '../../service/num';
import { ColorType, labelColor } from '../Painters/Painter';

type Section = number[];
type Stream = Section[];

interface OptionalProps {
  width: number;
  height: number;
  // min: number;
  // max: number;
  color: ColorType;
  transform?: string;
  display: boolean;
  // onMouseEnter?: React.MouseEventHandler<SVGGElement>;
  // onMouseLeave?: React.MouseEventHandler<SVGGElement>;
  // onClick?: React.MouseEventHandler<SVGGElement>;
}

export interface StreamPlotProps extends Partial<OptionalProps> {
  data: Stream;
  range?: [number, number];
}

export interface StreamPlotState {
}

export default class StreamPlot extends React.PureComponent<StreamPlotProps, StreamPlotState> {
  public static defaultProps: OptionalProps = {
    width: 100,
    height: 50,
    color: labelColor,
    display: true,
  };
  constructor(props: StreamPlotProps) {
    super(props);
  }

  render() {
    const { data, range, width, height, color, display, ...rest } = this.props as StreamPlotProps & OptionalProps;
    const stack = d3.stack<Section, number>();
    const streams = stack(data);
    const yMin = d3.min(streams, (stream) => d3.min(stream, (d) => d[0])) || -100;
    const yMax = d3.max(streams, (stream) => d3.max(stream, (d) => d[1])) || 100;
    
    const xScaler = d3.scaleLinear()
      .domain([-1, data.length]).range([0, width]);
    const yScaler = d3.scaleLinear()
      .domain([yMin, yMax]).range([0, height]);
    
    const area = d3.area<d3.SeriesPoint<number[]>>()
      .x((d, i) => xScaler(i))
      .y0((d, i) => yScaler(d[0]))
      .y1((d, i) => yScaler(d[1]));

    return (
      <g style={{display}} {...rest}>
        {streams.map((stream: d3.Series<number[], number>, i: number) => (
          <path d={area(stream) || undefined} fill={color(i)}/>
        ))}
        {range && 
        <rect x={width * range[0]} width={width * range[1] - range[0]} height={height}/>}
      </g>
    );
  }
}
