import UIKit
import Metal

fileprivate struct Const {
    // Gridサイズは 1,024 x 1,024 の１次元配列
    static let gridSize = threadGroupPerGrid * 32 * 32
    // threadsPerThreadGroupは1,024なので、Grid内のThread Groupの数は1,024
    static let threadGroupPerGrid = 1024
}

// 処理するデータの型
typealias DataType = Int32

class ViewController: UIViewController {

    @IBOutlet weak var label: UILabel!
    
    private let device = MTLCreateSystemDefaultDevice()!
    private lazy var commandQueue = device.makeCommandQueue()!
    private var computeSumPiplineState: MTLComputePipelineState!
    // 処理するデータ
    private var data: [DataType] = []
    // GPUに渡すデータのバッファ
    private var dataBuffer: MTLBuffer!
    // GPUの計算結果を受け取るバッファ
    private var resultsBuffer: MTLBuffer!
    private var results: UnsafeBufferPointer<DataType>!
    
    let serialQueue = DispatchQueue(label: "serial_queue", qos: .userInteractive, attributes: .concurrent, autoreleaseFrequency: .workItem, target: nil)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        makeRandomData()
        setupMetal()
        setupViews()
    }

    func setupViews() {
        label.text = ""
    }
    
    @IBAction func didTapStartButton(_ sender: Any) {
        label.text = ""
        
        compute()
    }

    func makeRandomData() {
        data = (0..<Const.gridSize).map{ _ in DataType.random(in: 0..<100) }
//        data = (0..<Const.gridSize).map{ _ in DataType(1) }   // 要素数の確認用
//        data = (0..<Const.gridSize).map{ _ in DataType(1.1) }   // 浮動小数点値の誤差の確認用
    }
    
    func setupMetal() {
        // コンピュートパイプライン生成
        let defaultLibrary = device.makeDefaultLibrary()!
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = defaultLibrary.makeFunction(name: "group_max")!
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        computeSumPiplineState = try! device.makeComputePipelineState(descriptor: descriptor,
                                                                      options: [],
                                                                      reflection: nil)
        // GPUとのデータ受け渡し用のバッファ確保
        dataBuffer = device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * Int(Const.gridSize), options: [])!
        resultsBuffer = device.makeBuffer(length: MemoryLayout<DataType>.stride * Const.threadGroupPerGrid, options: [])!
        let pointer = resultsBuffer.contents().bindMemory(to: DataType.self, capacity: Const.threadGroupPerGrid)
        results = UnsafeBufferPointer<DataType>(start: pointer, count: Const.threadGroupPerGrid)
    }
    
    func compute() {
//        DispatchQueue.main.async {
        serialQueue.async {
            let commandBuffer = self.commandQueue.makeCommandBuffer()!
            let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
            computeEncoder.setComputePipelineState(self.computeSumPiplineState)
            computeEncoder.setBuffer(self.dataBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(self.resultsBuffer, offset: 0, index: 1)
            
            let threadsPerThreadgroup = MTLSizeMake(self.computeSumPiplineState.maxTotalThreadsPerThreadgroup, 1, 1)
            let threadsPerGrid = MTLSizeMake(Const.gridSize, 1, 1)
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
            
            //
            // GPUでの計算開始
            //
            var start, cpuStart, end: UInt64
            var gpuResult: DataType = 0
            start = mach_absolute_time()
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            cpuStart = mach_absolute_time()
            for element in self.results {
                gpuResult += element
            }
            end = mach_absolute_time()
            
            let gpuCpuTime = Double(end - start) / Double(NSEC_PER_SEC) * 1000
            let cpuTime = Double(end - cpuStart) / Double(NSEC_PER_SEC) * 1000
            var resultLabel = "GPU:\n\n sum [\(gpuResult)]\n time[\(String(format: "%.7f", gpuCpuTime))]ms\n 内、CPU time[\(String(format: "%.7f", cpuTime))]ms\n\n"
//            var resultLabel = "GPU:\n\n sum [\(String(format: "%.1f", gpuResult))]\n time[\(String(format: "%.7f", gpuCpuTime))]s\n 内、CPU time[\(String(format: "%.7f", cpuTime))]s\n\n"

            //
            // CPUでの計算開始
            //
            var cpuResult: DataType = 0
            self.data.withUnsafeBufferPointer { buffer in
                // dataがメモリ上連続している状態で計測
                start = mach_absolute_time()
                for element in buffer {
                    cpuResult += element
                }
            }
            end = mach_absolute_time()
            
            let cupTime = Double(end - start) / Double(NSEC_PER_SEC) * 1000
            resultLabel += "CPU:\n\n sum [\(cpuResult)]\n time[\(String(format: "%.7f", cupTime))]ms\n\n\n"
            resultLabel += "sumの計算差異(CPU-GPU) [\(cpuResult - gpuResult)]\n\n"
            resultLabel += "GPUの計算速度は CPUだけの場合の[\(String(format: "%.1f", cupTime/gpuCpuTime))]倍"
            
            DispatchQueue.main.async {
                self.label.text = resultLabel
                print(resultLabel)
            }
        }
    }
}
