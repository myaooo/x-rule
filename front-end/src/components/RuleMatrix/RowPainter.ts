import * as d3 from 'd3';

import { ColorType, labelColor as defaultLabelColor, Painter } from '../Painters';
import { RuleX, ConditionX } from './models';
import { defaultDuration } from '../Painters/Painter';

// export interface RectGlyphProps {
//   // x: number;
//   width: number;
//   height: number;
//   // expand?: boolean;
//   // streams?: Stream; 
//   color?: ColorType;
//   range?: [number, number];
//   supports?: number[];
//   transform?: string;
//   style?: React.CSSProperties;
//   onClick?: () => void;
// }

// export class RectGlyph implements Painter<RectGlyphProps, any> {

//   render() {
//     const {width, height, range, supports, color, ...rest} = this.props;
//     let supportElements;
//     let rangeRect;
//     if (range) {
//       const x = width * range[0];
//       const rangeWidth = width * (range[1] - range[0]);
//       rangeRect = (<rect x={x} width={rangeWidth} height={height} className="mc-range"/>);
//       if (supports && color) {
//         // const totalSupports = nt.sum(supports);
//         const supportCumulated = nt.cumsum(supports);
//         const yScaler = d3.scaleLinear()
//           .domain([0, range[1] - range[0]])
//           .range([height, 0]);

//         const multiplier = height / (range[1] - range[0]);
//         supportElements = (
//           <g className="mc-satisfied" > 
//             {supports.map((support: number, i: number) => (
//               <rect
//                 key={i} 
//                 x={x}
//                 y={yScaler(supportCumulated[i])}
//                 width={rangeWidth} 
//                 height={support * multiplier}
//                 fill={color(i)}
//               />
//             ))}
//           </g>
//         );
//       }
//     }
//     return (
//       <g {...rest} className="matrix-condition">
//         <rect width={width} height={height} className="mc-bg"/>
//         {rangeRect}
//         {supportElements}
//       </g>
//     );
//   }
// }

export interface ConditionPainterParams {
  // stream?: Stream;
  // interval?: [number, number];
  // streamColor?: ColorType;
}

interface ConditionData {

}

export class ConditionPainter implements Painter<ConditionData, ConditionPainterParams> {
  private params: ConditionPainterParams;

  update(params: ConditionPainterParams): this {
    this.params = {...(this.params), ...params};
    return this;
  }
  data(newData: ConditionData): this {

    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGElement, ConditionData, GElement, any>,
  ): this {
    return this;
  }
}

interface OptionalParams {
  labelColor: ColorType;
  duration: number;
}

export interface RuleRowParams extends Partial<OptionalParams> {

  feature2Idx: (feature: number) => number;
  onClick?: (feature: number, condition: number) => void;
}

export default class RuleRowPainter implements Painter<RuleX, RuleRowParams> {
  public static defaultParams: OptionalParams = {
    labelColor: defaultLabelColor,
    duration: defaultDuration,
    // expandFactor: [4, 3],
  };
  private rule: RuleX;
  private params: RuleRowParams & OptionalParams;
  // constructor(props: RuleRowParams) {
  //   const {xs, widths} = props;
  //   const rowWidth = xs[xs.length - 1] + widths[widths.length - 1];
  // }
  
  update(params: RuleRowParams): this {
    this.params = {...(RuleRowPainter.defaultParams), ...(this.params), ...params};
    return this;
  }
  data(newData: RuleX): this {
    this.rule = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(
    selector: d3.Selection<SVGGElement, RuleX, GElement, any>,
  ): this {
    const { duration } = this.params;
    const rule = this.rule;

    // ROOT Group
    selector.selectAll('g.matrix-rule').data(['d']).enter()
      .append('g').attr('class', 'matrix-rule');
    const rootGroup = selector.select('g.matrix-rule');

    // Background Rectangle
    const bgRect = rootGroup.selectAll('rect.matrix-bg').data(['bg']);
    const bgRectUpdate = bgRect.enter()
      .append('rect').attr('class', 'matrix-bg').attr('width', 0).attr('height', 0)
      .merge(bgRect);
    bgRectUpdate.transition().duration(duration)
      .attr('width', rule.width).attr('height', rule.height);
    
    // JOIN
    const conditions = rootGroup.selectAll('g.matrix-condition').data(rule.conditions);

    // ENTER
    const conditionsEnter = conditions.enter()
      .append('g').attr('class', 'matrix-condition');

    // UPDATE
    const conditionsUpdate = conditionsEnter.merge(conditions);
    // Transition
    conditionsUpdate
      .transition().duration(duration)
      .attr('transform', (c: ConditionX) => `translate(${c.x}, 0)`);
    
    // EXIT
    conditions.exit().transition().duration(duration)
      .attr('transform', 'translate(0,0)').remove();
    return this;
  }
}
