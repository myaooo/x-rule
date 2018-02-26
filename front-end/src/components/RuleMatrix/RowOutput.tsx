import * as React from 'react';
import { ColorType } from '../Painters/Painter';
import * as nt from '../../service/num';

export interface RowOutputProps {
  outputWidth: number;
  outputs: number[];
  height: number;
  supports?: number[];  
  transform?: string;
  className?: string;
  color: ColorType;
}

export default class RowOutput extends React.Component<RowOutputProps, any> {
  render() {
    const {outputs, height, outputWidth, color, supports, ...rest} = this.props;
    const outputWidths = outputs.map((o) => o * outputWidth);
    const outputXs = [0, ...(nt.cumsum(outputWidths.slice(0, -1)))];
    // const getOutputTargetState = (o: number, i: number) => (
    //   {
    //     x: [outputXs[i] + i], 
    //     width: [outputWidths[i]], 
    //     height: [height / 4],
    //     timing: {delay: 0, duration: duration - 100},
    //   });
    // const totalWidth = nt.sum(widths);
    return (
      // <RectNodeGroup 
      //   data={outputs} 
      //   keyAccessor={(d, i) => i.toString()}
      //   start={(d, i) => ({x: 0, width: 0, height: 0})}
      //   enter={getOutputTargetState}
      //   update={getOutputTargetState}
      //   leave={(d, i) => ({x: 0, width: 0, height: 0})}
      // >
      //   {(nodes) => (
      <g {...rest}>
        {outputs.map((o: number, i: number) => {
          const state = {
            x: outputXs[i] + i,
            width: outputWidths[i],
            height
          };
          return (
            // <g key={i}>
              <rect 
                key={i}
                fill={color(i)}
                {...state}
              />
            // </g>
          );
        })}
      </g>
        // )}
      // </RectNodeGroup>
    );
  }
}
