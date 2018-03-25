// import * as React from 'react';
// import { Collapse } from 'antd';
// import { connect } from 'react-redux';
// import { RootState, getModel } from '../store';
// import DataTable from '../components/DataTable';
// import dataService from '../service/dataService';
// import { FilterType } from '../components/DataFilter';
// import { getSelectedData } from '../store/selectors';
// import { ModelBase, DataSet, rankModelFeatures } from '../models';
// import { changeFilterAndFetchData, Dispatch } from '../store/actions';

// const Panel = Collapse.Panel;

// // export interface DataInputHeaderProps {
// //   onClick?: () => void;
// // }

// // export function DataInputHeader(props: DataInputHeaderProps) {
// //   const { onClick } = props;
// //   return (
// //     <div>
// //       Data Filter
// //       <Button onClick={onClick} type="primary" icon="upload" 
// style={{ float: 'right', fontSize: 12 }} size="small">
// //         Filter
// //       </Button>
// //     </div>
// //   );
// // }

// export interface DataFilterProps {
//   model: ModelBase | null;
//   // meta?: ModelMeta;
//   // modelName: string;
//   dataSets: DataSet[];
//   changeData?: (filters: FilterType[]) => void;
// }

// export class DataFilter extends React.Component<DataFilterProps> {
//   private tableRef: DataTable;
//   constructor(props: DataFilterProps) {
//     super(props);
//     this.handleSubmitFilter = this.handleSubmitFilter.bind(this);
//   }
//   handleSubmitFilter() {
//     const changeData = this.props.changeData;
//     console.log('Clicked submit'); // tslint:disable-line
//     if (changeData) {
//       changeData(this.tableRef.state.filters);
//     }
//   }
//   render() {
//     const { model, dataSets } = this.props;
//     const width = 1200;
//     const height = 150;
//     const indices = model ? rankModelFeatures(model) : undefined;
//     const dataSet = dataSets[0] as (DataSet | undefined);
//     return (
//       <Collapse defaultActiveKey={[]} style={{position: 'fixed', bottom: 0}}>
//         <Panel header="Data View" key="1" style={{ width }}>
//           {model && (
//             <DataTable
//               dataSet={dataSet}
//               ref={(ref: DataTable) => this.tableRef = ref}
//               meta={model.meta}
//               height={height}
//               getData={(filters: FilterType[], start?: number, end?: number) =>
//                 dataService.getFilterData(model.name, dataSet ? dataSet.name : 'train', filters, start, end)
//               }
//               onSubmitFilter={this.handleSubmitFilter}
//               indices={indices}
//             />
//           )}
//         </Panel>
//       </Collapse>
//     );
//   }
// }

// const mapStateToProps = (state: RootState): DataFilterProps => {
//   return {
//     model: getModel(state),
//     dataSets: getSelectedData(state),
//     // modelName
//   };
// };

// const mapDispatchToProps =  (dispatch: Dispatch, ownProps: any): Partial<DataFilterProps> => {
//   return {
//     changeData: (filters: FilterType[]) => dispatch(changeFilterAndFetchData(filters))
//   };
// };

// export default connect(mapStateToProps, mapDispatchToProps)(DataFilter);
