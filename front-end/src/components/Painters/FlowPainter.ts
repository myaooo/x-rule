// import * as nt from '../../service/num';
// import { ColorType, Painter, labelColor, defaultDuration } from './index';

// // type Point = {x: number, y: number};
// // const originPoint = {x: 0, y: 0};

// // const curve = (s: Point = originPoint, t: Point = originPoint): string => {
// //   let dy = t.y - s.y;
// //   let dx = t.x - s.x;
// //   const r = Math.min(Math.abs(dx), Math.abs(dy));
// //   if (Math.abs(dx) > Math.abs(dy))
// //     return `M${s.x},${s.y} A${r},${r} 0 0 0 ${s.x + r} ${t.y} H ${t.x}`;
// //   else
// //     return `M ${s.x},${s.y} V ${s.y - r} A${r},${r} 0 0 0 ${t.x} ${t.y} `;
// // };

// // const flowCurve = (d?: {s: Point, t: Point}): string => {
// //   if (d) return curve(d.s, d.t);
// //   return curve();
// // };

// // function drawRects()

// interface FlowOptional {
//   width: number;
//   dx: number;
//   dy: number;
//   height: number;
//   duration: number;
//   color: ColorType;
// }

// interface FlowPainterParams extends Partial<FlowOptional> {}

// type Rect = {x: number, width: number, height: number};

// // type FlowData = { width: number; shift: number; height: number; y: number };

// type Flow = {support: number[], y: number};

// export default class FlowPainter implements Painter<Flow[], FlowPainterParams> {
//   public static defaultParams: FlowOptional = {
//     width: 100,
//     height: 50,
//     duration: defaultDuration,
//     dy: -30,
//     dx: -40,
//     color: labelColor
//     // fontSize: 12,
//     // multiplier: 1.0,
//   };
//   private params: FlowPainterParams & FlowOptional;
//   private flows: Flow[];
//   // private totalFlows: number[];
//   private outFlows: number[];
//   private reserves: Float32Array[];
//   private reserveSums: Float32Array;
//   public update(params: FlowPainterParams) {
//     this.params = { ...FlowPainter.defaultParams, ...this.params, ...params };
//     return this;
//   }
//   public data(flows: Flow[]) {
//     this.flows = flows;
//     this.outFlows = flows.map((r: Flow) => nt.sum(r.support));

//     let reserves: Float32Array[] = flows[0].support.map((_, i) => new Float32Array(flows.map(rule => rule.support[i])));
//     this.reserves = reserves.map((reserve: Float32Array) => nt.cumsum(reserve.reverse()).reverse());
//     this.reserveSums = new Float32Array(this.reserves[0].length);
//     this.reserves.forEach((reserve: Float32Array) => nt.add(this.reserveSums, reserve, false));
//     // console.log(this.reserves); // tslint:disable-line
//     // console.log(this.reserveSums); // tslint:disable-line
//     return this;
//   }
//   public render(selector: d3.Selection<SVGElement, any, any, any>): this {
//     const {width} = this.params;
//     // Make sure the root group exits
//     selector
//       .selectAll('g.flows')
//       .data(['flows'])
//       .enter()
//       .append('g')
//       .attr('class', 'flows');
//     const rootGroup = selector.select<SVGGElement>('g.flows')
//       .attr('transform', `translate(${-width}, 0)`);

//     // Render Rects
//     this.renderRects(rootGroup);
//     // // Join
//     // const rule = selector.select('g.flows')
//     //   .selectAll<SVGGElement, Flow>('g.flow')
//     //   .data(this.flows);
    
//     return this;
//   }

//   public renderRects(root: d3.Selection<SVGGElement, any, any, any>): this {
//     const {duration, height, width, dy} = this.params;
//     const {flows, reserves, reserveSums} = this;
//     // Compute pos
//     const heights = flows.map((f, i) => i > 0 ? f.y - flows[i - 1].y : height);
//     const multiplier = width / reserveSums[0];

//     // JOIN
//     const reserve = root.selectAll('v-reserve').data(flows);

//     // ENTER
//     const reserveEnter = reserve.enter().append('g').attr('class', 'v-reserve');

//     // UPDATE
//     const reserveUpdate = reserveEnter.merge(reserve);
//     // Transition groups
//     reserveUpdate.transition().duration(duration)
//       .attr('transform', (d: Flow, i: number) => `translate(0,${d.y - heights[i] - dy})`);
    
//     const rects = reserveUpdate.selectAll('rect')
//       .data<Rect>((d: Flow, i: number) => {
//         const widths = reserves[i].map((r) => r * multiplier);
//         const xs = [0, ...(nt.cumsum(widths.slice(0, -1)))];
//         return d.support.map((s: number, j: number) => {
//           return {
//             width: widths[j], height: heights[i], x: xs[j]
//           };
//         });
//       });
    
//     // RECT ENtER
//     const rectsEnter = rects.enter().append('rect');
      
//     // RECT UPDATE
//     const rectsUpdate = rectsEnter.merge(rects);
//     rectsUpdate.transition().duration(duration)
//       .attr('width', d => d.width).attr('height', d => d.height);

//     return this;
//   }

//   // private updatePos() {
//   //   const {outFlows, reserves, reserveSums} = this;
//   //   // const heights = ys.map
//   // }
// }