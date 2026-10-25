//go:build !windows

package apiserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Empty / whitespace-only input is a client error (400), not a server error
// (500). The OpenAPI contract documents {200, 400} for these endpoints, and
// sibling endpoints (combined, batch, embeddings) already return 400 for the
// identical condition.
func TestClassifyEmptyInputReturns400(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	cases := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{"intent empty", "/api/v1/diagnostics/classify/intent", `{"text":""}`, s.handleIntentClassification},
		{"pii empty", "/api/v1/diagnostics/classify/pii", `{"text":""}`, s.handlePIIDetection},
		{"security empty", "/api/v1/diagnostics/classify/security", `{"text":""}`, s.handleSecurityDetection},
		{"fact-check whitespace", "/api/v1/diagnostics/classify/fact-check", `{"text":" \t\n"}`, s.handleFactCheckClassification},
		{"user-feedback whitespace", "/api/v1/diagnostics/classify/user-feedback", `{"text":" \t\n"}`, s.handleUserFeedbackClassification},
		{"combined whitespace", "/api/v1/diagnostics/classify/combined", `{"text":" \t\n"}`, s.handleCombinedClassification},
		{"pii whitespace", "/api/v1/diagnostics/classify/pii", `{"text":"   "}`, s.handlePIIDetection},
		{"security whitespace", "/api/v1/diagnostics/classify/security", `{"text":"\t \n"}`, s.handleSecurityDetection},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			tc.handler(rr, req)
			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400 for %q, got %d: %s", tc.body, rr.Code, rr.Body.String())
			}
			if !strings.Contains(rr.Body.String(), "INVALID_INPUT") {
				t.Fatalf("expected INVALID_INPUT code, got: %s", rr.Body.String())
			}
		})
	}
}

// Non-empty input must still be served (placeholder service returns a graceful
// 200), i.e. the fix must not turn valid requests into 400s.
func TestClassifyNonEmptyInputStillOK(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/classify/pii", strings.NewReader(`{"text":"my email is a@b.com"}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	s.handlePIIDetection(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 for non-empty input, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestClassificationEndpointsMapRuntimeErrorsConsistently(t *testing.T) {
	errorCases := []struct {
		name           string
		err            error
		expectedStatus int
		expectedCode   string
	}{
		{
			name:           "invalid input",
			err:            fmt.Errorf("classification input: %w", binding.ErrInvalidInput),
			expectedStatus: http.StatusBadRequest,
			expectedCode:   "INVALID_INPUT",
		},
		{
			name:           "runtime rejected input",
			err:            fmt.Errorf("classifier runtime: %w", modelservice.ErrRejected),
			expectedStatus: http.StatusBadRequest,
			expectedCode:   "INVALID_INPUT",
		},
		{
			name:           "unsupported capability",
			err:            fmt.Errorf("classification backend: %w", binding.ErrCapability),
			expectedStatus: http.StatusConflict,
			expectedCode:   "UNSUPPORTED_CAPABILITY",
		},
		{
			name:           "probabilities unavailable",
			err:            fmt.Errorf("classification backend: %w", tasks.ErrProbabilitiesUnavailable),
			expectedStatus: http.StatusConflict,
			expectedCode:   "UNSUPPORTED_CAPABILITY",
		},
		{
			name:           "scan budget exceeded",
			err:            fmt.Errorf("classification input: %w", binding.ErrScanBudget),
			expectedStatus: http.StatusBadRequest,
			expectedCode:   "INVALID_INPUT",
		},
		{
			name:           "not prepared",
			err:            fmt.Errorf("classification backend: %w", binding.ErrNotPrepared),
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_NOT_READY",
		},
		{
			name:           "closed runtime",
			err:            fmt.Errorf("classification backend: %w", binding.ErrClosed),
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_NOT_READY",
		},
		{
			name:           "classifier unavailable",
			err:            fmt.Errorf("classification backend: %w", services.ErrClassifierUnavailable),
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_UNAVAILABLE",
		},
		{
			name:           "built-in model runtime unavailable",
			err:            fmt.Errorf("classifier runtime: %w", modelservice.ErrUnavailable),
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_NOT_READY",
		},
		{
			name:           "runtime readiness wait timed out",
			err:            fmt.Errorf("%w: %w", modelservice.ErrUnavailable, context.DeadlineExceeded),
			expectedStatus: http.StatusGatewayTimeout,
			expectedCode:   "CLASSIFICATION_TIMEOUT",
		},
		{
			name:           "model runtime overloaded",
			err:            fmt.Errorf("classifier runtime: %w", modelservice.ErrOverloaded),
			expectedStatus: http.StatusTooManyRequests,
			expectedCode:   "OVERLOADED",
		},
		{
			name:           "timeout",
			err:            fmt.Errorf("classification timed out: %w", context.DeadlineExceeded),
			expectedStatus: http.StatusGatewayTimeout,
			expectedCode:   "CLASSIFICATION_TIMEOUT",
		},
		{
			name:           "internal",
			err:            errors.New("inference runtime failure"),
			expectedStatus: http.StatusInternalServerError,
			expectedCode:   "CLASSIFICATION_ERROR",
		},
	}
	endpoints := []struct {
		name    string
		path    string
		body    string
		handler func(*ClassificationAPIServer) func(http.ResponseWriter, *http.Request)
		inject  func(*evalCaptureClassificationService, error)
	}{
		{
			name: "intent", path: "/api/v1/diagnostics/classify/intent", body: `{"text":"hello"}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handleIntentClassification
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.intentErr = err },
		},
		{
			name: "pii", path: "/api/v1/diagnostics/classify/pii", body: `{"text":"hello"}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handlePIIDetection
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.piiErr = err },
		},
		{
			name: "security", path: "/api/v1/diagnostics/classify/security", body: `{"text":"hello"}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handleSecurityDetection
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.securityErr = err },
		},
		{
			name: "batch", path: "/api/v1/diagnostics/classify/batch", body: `{"texts":["hello"]}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handleBatchClassification
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.batchErr = err },
		},
		{
			name: "fact-check", path: "/api/v1/diagnostics/classify/fact-check", body: `{"text":"hello"}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handleFactCheckClassification
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.factCheckErr = err },
		},
		{
			name: "user-feedback", path: "/api/v1/diagnostics/classify/user-feedback", body: `{"text":"hello"}`,
			handler: func(s *ClassificationAPIServer) func(http.ResponseWriter, *http.Request) {
				return s.handleUserFeedbackClassification
			},
			inject: func(s *evalCaptureClassificationService, err error) { s.feedbackErr = err },
		},
	}

	for _, endpoint := range endpoints {
		for _, errorCase := range errorCases {
			t.Run(endpoint.name+"/"+errorCase.name, func(t *testing.T) {
				service := &evalCaptureClassificationService{}
				endpoint.inject(service, errorCase.err)
				server := &ClassificationAPIServer{classificationSvc: service}
				request := httptest.NewRequest(
					http.MethodPost,
					endpoint.path,
					strings.NewReader(endpoint.body),
				)
				request.Header.Set("Content-Type", "application/json")
				recorder := httptest.NewRecorder()
				endpoint.handler(server)(recorder, request)

				if recorder.Code != errorCase.expectedStatus {
					t.Fatalf("status = %d, want %d: %s", recorder.Code, errorCase.expectedStatus, recorder.Body.String())
				}
				if !strings.Contains(recorder.Body.String(), errorCase.expectedCode) {
					t.Fatalf("response = %s, want code %q", recorder.Body.String(), errorCase.expectedCode)
				}
			})
		}
	}

	combinedStages := []struct {
		name   string
		inject func(*evalCaptureClassificationService, error)
	}{
		{name: "intent", inject: func(s *evalCaptureClassificationService, err error) { s.intentErr = err }},
		{name: "pii", inject: func(s *evalCaptureClassificationService, err error) { s.piiErr = err }},
		{name: "security", inject: func(s *evalCaptureClassificationService, err error) { s.securityErr = err }},
	}
	for _, stage := range combinedStages {
		for _, errorCase := range errorCases {
			t.Run("combined/"+stage.name+"/"+errorCase.name, func(t *testing.T) {
				service := &evalCaptureClassificationService{}
				stage.inject(service, errorCase.err)
				server := &ClassificationAPIServer{classificationSvc: service}
				request := httptest.NewRequest(
					http.MethodPost,
					"/api/v1/diagnostics/classify/combined",
					strings.NewReader(`{"text":"hello"}`),
				)
				request.Header.Set("Content-Type", "application/json")
				recorder := httptest.NewRecorder()
				server.handleCombinedClassification(recorder, request)

				if recorder.Code != errorCase.expectedStatus {
					t.Fatalf("status = %d, want %d: %s", recorder.Code, errorCase.expectedStatus, recorder.Body.String())
				}
				if !strings.Contains(recorder.Body.String(), errorCase.expectedCode) {
					t.Fatalf("response = %s, want code %q", recorder.Body.String(), errorCase.expectedCode)
				}
			})
		}
	}
}
