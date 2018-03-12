import * as React from 'react';
import * as d3 from 'd3';
import { ColorType } from '../Painters/Painter';
import * as nt from '../../service/num';

function isMatSupport(supports: number[] | number[][]): supports is number[][] {
  return Array.isArray(supports[0]);
}

export interface SimpleSupportProps {
  supports: number[];
  width: number;
  height: number;
  color: ColorType;
  transform?: string;
  className?: string;
}

export class SimpleSupport extends React.PureComponent<SimpleSupportProps, any> {
  render() {
    const {supports, width, height, color, ...rest} = this.props;
    const widths = supports.map((s) => s * width);
    const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
    return (
      <g {...rest}>
        {supports.map((_, i) => (
          <rect key={i} x={xs[i] + i} width={widths[i]} height={height} fill={color(i)}/>
        ))}
      </g>
    );
  }
}

export interface MatSupportProps {
  supports: number[][];
  width: number;
  height: number;
  color: ColorType;
  transform?: string;
  className?: string;
}

export class MatSupport extends React.PureComponent<MatSupportProps, any> {
  render() {
    const {supports, width, height, color, ...rest} = this.props;
    const actualLabelCounts = nt.sumVec(supports);
    const predictLabelCounts = d3.transpose<number>(supports);
    const TPCounts = supports.map((s, i) => s[i]); // diag
    const TPHeights = TPCounts.map((c, i) => c / actualLabelCounts[i] * height);

    const widths = actualLabelCounts.map((s) => s * width);
    const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
    return (
      <g {...rest}>
        {predictLabelCounts.map((predicts, trueLabel: number) => {
          if (actualLabelCounts[trueLabel] < 1e-6) return <g key={trueLabel}/>;
          const x = xs[trueLabel] + trueLabel;
          const labelWidth = widths[trueLabel];
          const hasFalse = (actualLabelCounts[trueLabel] - TPCounts[trueLabel]);
          const trueRect = <rect key={trueLabel} x={x} width={labelWidth} height={height} fill={color(trueLabel)}/>;
          if (!hasFalse) return <g key={trueLabel}>{trueRect}</g>;
          const multiplier = labelWidth / hasFalse;
          const falseWidths = predicts.map((p: number, i: number) => i === trueLabel ? 0 : p * multiplier);
          const falseXs = [0, ...(nt.cumsum(falseWidths.slice(0, -1)))];
          const trueH = TPHeights[trueLabel];
          const falseH = height - trueH;
          return (
            <g key={trueLabel}>
              {trueRect}
              {predicts.map((s, i) => {
                return i === trueLabel
                  ? ''
                  : <rect 
                    key={i} 
                    x={x + falseXs[i]} 
                    y={trueH} 
                    width={falseWidths[i]} 
                    height={falseH} 
                    fill={color(i)}
                    className="mc-stripe"
                  />;
              })}

            </g>
          );
        })}
      </g>
    );
  }
}

export interface RowOutputProps {
  supportWidth: number;
  outputs: number[];
  height: number;
  supports?: number[] | number[][];  
  transform?: string;
  className?: string;
  color: ColorType;
}

// export function outputs2Confidence(outputs: number[]) {
//   if (outputs.length === 0) return -1;
//   const max = outputs[0];
// }

export default class RowOutput extends React.PureComponent<RowOutputProps, any> {
  render() {
    const {outputs, height, supportWidth, color, supports, ...rest} = this.props;
    const numberWidth = 26;
    const outputWidths = outputs.map((o) => o * numberWidth);
    const outputXs = [0, ...(nt.cumsum(outputWidths.slice(0, -1)))];
    const predictLabel = nt.argMax(outputs);
    const confidence = outputs[predictLabel];
    const confColor = d3.interpolateRgb.gamma(2.2)('#ddd', color(predictLabel))(confidence * 2 - 1);
    // const getOutputTargetState = (o: number, i: number) => (
    //   {
    //     x: [outputXs[i] + i], 
    //     width: [outputWidths[i]], 
    //     height: [height / 4],
    //     timing: {delay: 0, duration: duration - 100},
    //   });
    // const totalWidth = nt.sum(widths);
    const supportProps = {
      width: supportWidth, height, transform: `translate(40, 0)`, color
    };
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
        <text fill={confColor} textAnchor="start" dy="12">
          {Math.round(confidence * 100) / 100}
        </text>
        <g>
          {// outputs
            outputs.map((o: number, i: number) => {
              const state = {
                x: outputXs[i] + i,
                y: 15,
                width: outputWidths[i],
                height: 4
              };
              return (
                <rect key={i} fill={color(i)} {...state}/>
              );
            })
          }
        </g>
        {supports && (
          isMatSupport(supports)
          ? <MatSupport supports={supports} {...supportProps}/>
          : <SimpleSupport supports={supports} {...supportProps}/>
        )
        }
      </g>
        // )}
      // </RectNodeGroup>
    );
  }
}
